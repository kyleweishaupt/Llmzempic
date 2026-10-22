"""Routes for project packages: download a project as a .llmz file, import one, duplicate a project.
A large package takes a while to build, so it is prepared first (the page can show that) and then
downloaded once by its token."""

import re
import time
import uuid

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .packages import PackageError, duplicate_project, export_package, import_package

TOKEN = re.compile(r"[0-9a-f]{32}")
ABANDONED_SECONDS = 3600


def package_router(ws, project_or_404, project_view):
	router = APIRouter()

	def temp_path():
		return ws.root / f".package-{uuid.uuid4().hex}.llmz"

	@router.post("/api/projects/{project_id}/package")
	async def prepare_package(project_id: str, payload: dict = Body(default={})):
		project = project_or_404(project_id)
		for old in ws.root.glob(".package-*.llmz"):  # prepared but never downloaded
			if time.time() - old.stat().st_mtime > ABANDONED_SECONDS:
				old.unlink(missing_ok=True)
		token = uuid.uuid4().hex
		path = ws.root / f".package-{token}-{project.id}.llmz"
		await run_in_threadpool(export_package, ws, project, path, bool(payload.get("include_run")))
		return {"token": token, "filename": f"{project.id}.llmz", "size": path.stat().st_size}

	@router.get("/api/packages/{token}")
	def download_package(token: str):
		found = list(ws.root.glob(f".package-{token}-*.llmz")) if TOKEN.fullmatch(token) else []
		if not found:
			raise HTTPException(404, "That package isn't ready or was already downloaded. Export it again.")
		path = found[0]
		project_id = path.name[len(f".package-{token}-"):-len(".llmz")]
		return FileResponse(
			path, media_type="application/zip", filename=f"{project_id}.llmz",
			background=BackgroundTask(path.unlink, missing_ok=True),
		)

	@router.post("/api/packages")
	async def upload_package(request: Request):
		path = temp_path()
		try:
			with open(path, "wb") as f:
				async for chunk in request.stream():
					f.write(chunk)
			project = await run_in_threadpool(import_package, ws, path)
		except PackageError as e:
			raise HTTPException(400, str(e))
		finally:
			path.unlink(missing_ok=True)
		return project_view(project)

	@router.post("/api/projects/{project_id}/duplicate")
	def duplicate(project_id: str):
		return project_view(duplicate_project(ws, project_or_404(project_id)))

	return router
