"""Routes for example images: upload into a project, serve project and run snapshots."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .images import ImageError, image_path, media_type, save_image

# Names are content hashes, so a URL's bytes never change.
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def image_response(folder, name):
	try:
		path = image_path(folder, name)
	except KeyError:
		raise HTTPException(404, "Image not found")
	return FileResponse(path, media_type=media_type(name), headers=IMMUTABLE)


def image_router(project_or_404, run_or_404):
	router = APIRouter()

	@router.post("/api/projects/{project_id}/images")
	async def upload_image(project_id: str, request: Request):
		"""The body is the raw image file (no multipart), so no extra dependency is needed."""
		project = project_or_404(project_id)
		try:
			name = save_image(project.images_dir, await request.body())
		except ImageError as e:
			raise HTTPException(400, str(e))
		return {"name": name}

	@router.get("/api/projects/{project_id}/images/{name}")
	def project_image(project_id: str, name: str):
		return image_response(project_or_404(project_id).images_dir, name)

	@router.get("/api/projects/{project_id}/runs/{run_id}/images/{name}")
	def run_image(project_id: str, run_id: str, name: str):
		return image_response(run_or_404(project_or_404(project_id), run_id) / "images", name)

	return router
