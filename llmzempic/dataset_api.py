"""Routes for the workspace's reference data set library: preview, streaming upload and replace,
search for the value picker, and delete (refused while a project's schema uses the set)."""

import uuid

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from .datasets import DataSetError, preview
from .refs import referenced_ids


def dataset_router(ws, checker):
	router = APIRouter()

	def users(dataset_id):
		return [p for p in ws.projects() if dataset_id in referenced_ids(p.load_config().get("schema"))]

	def recheck(dataset_id):
		for project in users(dataset_id):
			checker.ensure(project, ws.library)

	def dataset_or_404(dataset_id):
		try:
			return ws.library.get(dataset_id)
		except KeyError:
			raise HTTPException(404, "Data set not found")

	async def receive(request, build):
		"""Stream the raw body to a temp file, then build the data set from it off the event loop."""
		ws.root.mkdir(parents=True, exist_ok=True)
		temp = ws.root / f".upload-{uuid.uuid4().hex}.tmp"
		try:
			with open(temp, "wb") as f:
				async for chunk in request.stream():
					f.write(chunk)
			return await run_in_threadpool(build, temp)
		except DataSetError as e:
			raise HTTPException(400, str(e))
		finally:
			temp.unlink(missing_ok=True)

	@router.get("/api/datasets")
	def list_datasets():
		return {"datasets": [{**m, "used_by": [p.load_config().get("name") or p.id for p in users(m["id"])]} for m in ws.library.list()]}

	@router.post("/api/datasets/preview")
	def preview_dataset(payload: dict = Body(...)):
		try:
			return preview(
				str(payload.get("text") or ""), str(payload.get("filename") or ""),
				payload.get("value_column") or None, payload.get("description_column"),
			)
		except DataSetError as e:
			raise HTTPException(400, str(e))

	@router.post("/api/datasets")
	async def upload_dataset(request: Request, name: str, filename: str, value_column: str = None, description_column: str = None):
		return await receive(request, lambda path: ws.library.create(name, path, filename, value_column, description_column))

	@router.put("/api/datasets/{dataset_id}")
	async def replace_dataset(dataset_id: str, request: Request, filename: str, value_column: str = None, description_column: str = None):
		dataset_or_404(dataset_id)
		meta = await receive(request, lambda path: ws.library.replace(dataset_id, path, filename, value_column, description_column))
		recheck(dataset_id)
		return meta

	@router.delete("/api/datasets/{dataset_id}")
	def delete_dataset(dataset_id: str):
		dataset_or_404(dataset_id)
		names = [p.load_config().get("name") or p.id for p in users(dataset_id)]
		if names:
			raise HTTPException(409, f"Used by: {', '.join(names)}. Remove it from their schemas first.")
		ws.library.delete(dataset_id)
		return {"ok": True}

	@router.get("/api/datasets/{dataset_id}/search")
	def search_dataset(dataset_id: str, q: str = "", limit: int = 20):
		return {"rows": dataset_or_404(dataset_id).search(q, min(max(1, limit), 100))}

	return router
