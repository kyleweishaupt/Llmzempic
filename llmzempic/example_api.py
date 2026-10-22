"""Routes for a project's examples: server-side paging and filters, single edits checked on the spot,
bulk operations with undo, streaming import of files of any size, and JSONL export."""

import json
import os
import uuid

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from .checks import stats
from .data import iter_records
from .example_store import Filter
from .schema_tools import infer_schema

MAX_WARNINGS = 50
IMPORT_MODES = ("append", "replace", "update")


def _filter(kind="all", search="", field=None, value=None, label_kind=None):
	label = None
	if field:
		try:
			parsed = json.loads(value) if value is not None else None
		except ValueError:
			parsed = value  # a bare string such as ?value=billing
		label = {"field": field, "value": parsed, "kind": label_kind or "enum"}
	return Filter(kind, search, label)


def _filter_from_body(data):
	data = data or {}
	return Filter(data.get("kind", "all"), data.get("search", ""), data.get("label"))


def example_router(ws, project_or_404, checker):
	router = APIRouter()

	def listing(project, flt, offset, limit, with_stats=True):
		checker.ensure(project, ws.library)
		store = project.examples()
		# Progress first: once it says checking is done, the rows read below already carry their errors.
		checking = checker.progress(project.id)
		view = {
			"examples": store.page(flt, offset, limit),
			"total": store.count(flt),
			"counts": {"all": store.count(), "problems": store.problem_count(), "duplicates": store.count(Filter("duplicates"))},
			"checking": checking,
		}
		if with_stats:
			view["stats"] = stats(project, ws.library, project.load_config())
		return view

	@router.get("/api/projects/{project_id}/examples")
	def get_examples(project_id: str, filter: str = "all", search: str = "", field: str = None, value: str = None,
			kind: str = None, offset: int = 0, limit: int = 20, include_stats: int = Query(1, alias="stats")):
		project = project_or_404(project_id)
		flt = _filter(filter, search, field, value, kind)
		return listing(project, flt, max(0, offset), min(max(1, limit), 500), bool(include_stats))

	@router.post("/api/projects/{project_id}/examples")
	def add_examples(project_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		rows = payload.get("examples")
		if not isinstance(rows, list):
			raise HTTPException(400, "Expected a list of examples.")
		added = project.examples().add(rows)
		errors = checker.check_now(project, ws.library, [r["id"] for r in added])
		return {"examples": [{**r, "errors": errors[r["id"]]} for r in added]}

	@router.put("/api/projects/{project_id}/examples")
	def replace_examples(project_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		rows = payload.get("examples")
		if not isinstance(rows, list):
			raise HTTPException(400, "Expected a list of examples.")
		project.examples().import_records(rows, "replace")
		checker.ensure(project, ws.library, force=True)
		return listing(project, Filter(), 0, 20)

	@router.put("/api/projects/{project_id}/examples/{example_id}")
	def update_example(project_id: str, example_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		try:
			project.examples().update(example_id, payload)
		except KeyError:
			raise HTTPException(404, "Example not found")
		errors = checker.check_now(project, ws.library, [example_id])
		return {**project.examples().get(example_id), "errors": errors[example_id]}

	@router.post("/api/projects/{project_id}/examples/bulk")
	def bulk(project_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		store = project.examples()
		ids = payload.get("ids")
		flt = None if ids is not None else _filter_from_body(payload.get("filter"))
		if payload.get("op") == "delete":
			op, n = store.delete(ids=ids, flt=flt)
		elif payload.get("op") == "set_output":
			op, n = store.set_output(payload.get("output"), ids=ids, flt=flt)
			checker.ensure(project, ws.library, force=True)
		else:
			raise HTTPException(400, "op must be delete or set_output.")
		return {"affected": n, "undo": op}

	@router.post("/api/projects/{project_id}/examples/undo")
	def undo(project_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		restored = project.examples().undo(str(payload.get("op") or ""))
		checker.ensure(project, ws.library, force=True)
		return {"restored": restored}

	@router.post("/api/projects/{project_id}/import")
	async def import_examples(project_id: str, request: Request, filename: str = "", mode: str = "append"):
		"""The body is the raw file, streamed to disk first, so any size works."""
		project = project_or_404(project_id)
		if mode not in IMPORT_MODES:
			raise HTTPException(400, f"mode must be one of: {', '.join(IMPORT_MODES)}")
		temp = project.path / f".import-{uuid.uuid4().hex}.tmp"
		try:
			with open(temp, "wb") as f:
				async for chunk in request.stream():
					f.write(chunk)
			result = await run_in_threadpool(_import_file, project, temp, filename, mode)
		finally:
			temp.unlink(missing_ok=True)
		if not result["added"] and not result["updated"]:
			raise HTTPException(400, "; ".join(result["warnings"][:5]) or "No examples found in the file.")
		checker.ensure(project, ws.library, force=True)
		return result

	@router.get("/api/projects/{project_id}/examples/export")
	def export_examples(project_id: str):
		project = project_or_404(project_id)

		def lines():
			for row in project.examples().iter_rows():
				row.pop("errors", None)
				yield json.dumps(row, ensure_ascii=False) + "\n"

		headers = {"Content-Disposition": f'attachment; filename="{project.id}-examples.jsonl"'}
		return StreamingResponse(lines(), media_type="application/x-ndjson", headers=headers)

	@router.post("/api/projects/{project_id}/infer-schema")
	def infer(project_id: str):
		outputs = project_or_404(project_id).examples().outputs(5000)
		if not outputs:
			raise HTTPException(400, "Add some examples first; the schema is inferred from their outputs.")
		return {"schema": infer_schema(outputs)}

	return router


def _import_file(project, path, filename, mode):
	schema = project.load_config()["schema"]
	name = os.path.basename(filename)
	if not any(example is not None for _, example, _ in iter_records(path, name, schema)):
		# Nothing usable: report the problems without touching (or, for "replace", wiping) the examples.
		return {"added": 0, "updated": 0, **_warnings_only(path, name, schema)}
	warnings, warning_count = [], 0

	def records():
		nonlocal warning_count
		for _, example, warning in iter_records(path, name, schema):
			if example is not None:
				yield example
			else:
				warning_count += 1
				if len(warnings) < MAX_WARNINGS:
					warnings.append(warning)

	counts = project.examples().import_records(records(), mode)
	return {**counts, "warning_count": warning_count, "warnings": warnings}


def _warnings_only(path, name, schema):
	warnings = [w for _, example, w in iter_records(path, name, schema) if example is None]
	return {"warning_count": len(warnings), "warnings": warnings[:MAX_WARNINGS]}
