"""Labeling helpers: importing a Hugging Face dataset, and generating synthetic examples with a stronger LLM."""

from fastapi import APIRouter, Body, HTTPException

from .hub_import import HubError, describe, import_examples, labels_schema
from .synthetic import SynthesisError, SynthesisJobs, generate


def assist_router(ws, project_or_404, checker, project_view):
	router = APIRouter()
	jobs = SynthesisJobs()

	def token():
		return ws.load_settings().get("hf_token") or None

	@router.get("/api/hub/dataset")
	def hub_dataset(id: str):
		if not id.strip():
			raise HTTPException(400, "Enter a dataset id, like mteb/banking77.")
		try:
			return describe(id.strip(), token())
		except HubError as e:
			raise HTTPException(400, str(e))

	@router.post("/api/projects/{project_id}/hub-import")
	def hub_import(project_id: str, payload: dict = Body(...)):
		"""Import a split's rows as examples; with set_schema, the output becomes one field per label column."""
		project = project_or_404(project_id)
		labels = payload.get("labels") or []
		if not payload.get("text") or not isinstance(labels, list) or not labels:
			raise HTTPException(400, "Choose the text column and at least one label column.")
		try:
			examples, seen, skipped, features = import_examples(
				payload.get("dataset", ""), payload.get("config", ""), payload.get("split", ""), payload["text"], labels,
				int(payload.get("limit") or 5000), token())
		except (HubError, ValueError) as e:
			raise HTTPException(400, str(e))
		if not examples:
			raise HTTPException(400, "No rows had both a text and every label.")
		if payload.get("set_schema"):
			project.save_config({"schema": labels_schema(seen, features)})
		mode = "replace" if payload.get("mode") == "replace" else "append"
		result = project.examples().import_records(examples, mode)
		checker.ensure(project, ws.library, force=True)
		return {**result, "skipped": skipped, "view": project_view(project)}

	@router.get("/api/projects/{project_id}/synthesize")
	def synthesis_status(project_id: str):
		project_or_404(project_id)
		status = jobs.status(project_id)
		status.pop("stop", None)
		return status

	@router.post("/api/projects/{project_id}/synthesize")
	def synthesize(project_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		config = project.load_config()
		if config.get("input_kind") == "image":
			raise HTTPException(400, "Synthetic examples are text only.")
		settings = ws.load_settings()
		if not settings.get("generator_url") or not settings.get("generator_model"):
			raise HTTPException(400, "Set the generator's URL and model in Settings first.")
		samples = [{"input": r["input"], "output": r["output"]} for r, _ in zip(project.examples().iter_rows(), range(3))]
		fields = payload.get("fields") or None

		def run(update, stopped):
			added = 0

			def on_batch(batch, done, total):
				nonlocal added
				if batch:
					rows = project.examples().add(batch)
					checker.check_now(project, ws.library, [r["id"] for r in rows])
					added += len(rows)
				update(done=done, total=total, added=added)

			generate(settings, config.get("instructions", ""), config["schema"], payload.get("per_value") or 5, samples, fields, on_batch, stopped)

		try:
			jobs.start(project.id, run)
		except SynthesisError as e:
			raise HTTPException(409, str(e))
		return {"ok": True}

	@router.delete("/api/projects/{project_id}/synthesize")
	def stop_synthesis(project_id: str):
		project_or_404(project_id)
		jobs.stop(project_id)
		return {"ok": True}

	return router
