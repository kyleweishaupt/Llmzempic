"""Local web UI + JSON API. Binds to localhost by default."""

import os
import shutil
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, hardware
from .checks import Checker
from .data import InvalidExample
from .dataset_api import dataset_router
from .example_api import example_router
from .package_api import package_router
from .fileio import read_json, tail_text
from .image_api import image_router
from .jobs import JobBusy, JobManager, job_paths, mark_interrupted
from .datasets import run_resolver
from .metrics import label_breakdown
from .ref_metrics import ref_scores
from .prompts import build_system_prompt
from .runs import (
	GGUF_TYPES, RunError, create_run, list_export_files, model_name, prepare_export, prepare_prediction,
	read_events, run_summary,
)
from .refs import export_contract, referenced_ids
from .schema_tools import TEMPLATES, check_schema
from .snippets import all_snippets
from .workspace import PRESETS, Workspace, load_catalog

WEB_DIR = Path(__file__).with_name("web")


class WebFiles(StaticFiles):
	"""Revalidate on every load (a cheap 304 on localhost) so an updated UI never mixes with stale cached modules."""

	def file_response(self, *args, **kwargs):
		response = super().file_response(*args, **kwargs)
		response.headers["Cache-Control"] = "no-cache"
		return response


class HardwareCache:
	def __init__(self):
		self._lock = threading.Lock()
		self._info = None

	def get(self, refresh=False):
		with self._lock:
			if self._info is None or refresh:
				self._info = hardware.detect()
			return self._info


def cross_site(request):
	if request.headers.get("sec-fetch-site") == "cross-site":
		return True
	origin = request.headers.get("origin")
	return origin is not None and urlsplit(origin).netloc != request.headers.get("host")


def create_app(workspace=None, jobs=None, hw_cache=None):
	ws = workspace or Workspace()
	jobs = jobs or JobManager()
	hw_cache = hw_cache or HardwareCache()
	checker = Checker()
	mark_interrupted(ws)
	# Warm the hardware probe in the background; the first page load would otherwise wait on it.
	threading.Thread(target=hw_cache.get, daemon=True).start()

	@asynccontextmanager
	async def lifespan(app):
		yield
		jobs.stop()

	app = FastAPI(title="Llmzempic", version=__version__, lifespan=lifespan)
	app.state.workspace = ws
	app.state.jobs = jobs

	@app.exception_handler(InvalidExample)
	async def invalid_example(request, error):
		return JSONResponse({"detail": str(error)}, status_code=400)

	@app.middleware("http")
	async def same_site_only(request, call_next):
		# There is no login, and a page on another site can send "simple" cross-origin POSTs (raw
		# uploads included) to localhost. Only this app's own pages may change anything.
		if request.method not in ("GET", "HEAD", "OPTIONS") and cross_site(request):
			return JSONResponse({"detail": "Requests from other sites are refused."}, status_code=403)
		return await call_next(request)

	def project_or_404(project_id):
		try:
			return ws.get(project_id)
		except KeyError:
			raise HTTPException(404, "Project not found")

	def run_or_404(project, run_id):
		try:
			return project.run_path(run_id)
		except KeyError:
			raise HTTPException(404, "Run not found")

	def job_env():
		token = ws.load_settings().get("hf_token")
		return {"HF_TOKEN": token} if token else {}

	def active_job():
		job = jobs.current()
		return job.describe() if job else None

	def project_view(project):
		config = project.load_config()
		names = {m["id"]: m["name"] for m in ws.library.list()}
		missing = sorted(i for i in referenced_ids(config["schema"]) if i not in names)
		return {
			"id": project.id,
			"config": config,
			"schema_errors": check_schema(config["schema"]) + [f'Data set "{i}" no longer exists.' for i in missing],
			"system_prompt": build_system_prompt(
				config.get("instructions", ""), config["schema"], config.get("include_schema_in_prompt", True), names
			),
		}

	@app.get("/api/system")
	def system(refresh: bool = False):
		return {
			"version": __version__,
			"workspace": str(ws.root),
			"hardware": hw_cache.get(refresh),
			"active_job": active_job(),
		}

	@app.get("/api/catalog")
	def catalog():
		hw = hw_cache.get()
		models = []
		for model in load_catalog():
			fit, detail = hardware.model_fit(model["params_b"], hw)
			models.append({**model, "fit": fit, "fit_detail": detail})
		return {"models": models, "templates": TEMPLATES, "presets": PRESETS, "gguf_types": GGUF_TYPES}

	@app.get("/api/settings")
	def get_settings():
		settings = ws.load_settings()
		return {"hf_token_set": bool(settings.get("hf_token")), "llama_cpp_dir": settings.get("llama_cpp_dir", "")}

	@app.put("/api/settings")
	def put_settings(payload: dict = Body(...)):
		values = {"llama_cpp_dir": payload.get("llama_cpp_dir")}
		if payload.get("hf_token") is not None:
			values["hf_token"] = str(payload["hf_token"]).strip()
		ws.save_settings(values)
		return get_settings()

	@app.get("/api/projects")
	def list_projects():
		items = []
		for project in ws.projects():
			config = project.load_config()
			runs = project.run_ids()
			last = run_summary(project.runs_dir / runs[0]) if runs else None
			items.append({
				"id": project.id,
				"name": config.get("name") or project.id,
				"base_model": config.get("base_model"),
				"input_kind": config.get("input_kind", "text"),
				"updated": config.get("updated") or config.get("created"),
				"last_run": last,
			})
		return {"projects": items}

	@app.post("/api/projects")
	def new_project(payload: dict = Body(...)):
		name = str(payload.get("name") or "").strip()
		if not name:
			raise HTTPException(400, "Give the project a name.")
		project = ws.create(name, payload.get("template") or "classification")
		return project_view(project)

	@app.get("/api/projects/{project_id}")
	def get_project(project_id: str):
		return project_view(project_or_404(project_id))

	@app.put("/api/projects/{project_id}")
	def put_project(project_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		config = payload.get("config") or {}
		if "schema" in config and not isinstance(config["schema"], dict):
			raise HTTPException(400, "The schema must be a JSON object.")
		project.save_config(config)
		return project_view(project)

	@app.delete("/api/projects/{project_id}")
	def delete_project(project_id: str):
		project = project_or_404(project_id)
		job = jobs.current()
		if job and job.run_dir.parent.parent == project.path:
			raise HTTPException(409, "Stop the running job before deleting this project.")
		ws.delete(project.id)
		return {"ok": True}

	@app.get("/api/projects/{project_id}/runs")
	def list_runs(project_id: str):
		project = project_or_404(project_id)
		return {"runs": [run_summary(project.runs_dir / r) for r in project.run_ids()], "active_job": active_job()}

	@app.post("/api/projects/{project_id}/runs")
	def start_run(project_id: str):
		project = project_or_404(project_id)
		if jobs.current():
			raise HTTPException(409, "Another job is running. Wait for it to finish or stop it.")
		hw = hw_cache.get()
		if not hw["can_train"]:
			missing = ", ".join(hw["missing_train_packages"]) or "PyTorch"
			raise HTTPException(400, f"Training libraries are missing ({missing}). Run the setup script, then restart.")
		if project.load_config().get("input_kind") == "image" and not hw["can_train_images"]:
			missing = ", ".join(hw["missing_image_packages"])
			raise HTTPException(400, f"Image projects need {missing}. Run the setup script, then restart.")
		try:
			run_dir = create_run(project, ws.library, checker)
		except RunError as e:
			raise HTTPException(400, str(e))
		try:
			jobs.start("train", run_dir, job_env())
		except JobBusy as e:
			raise HTTPException(409, str(e))
		return run_summary(run_dir)

	@app.get("/api/projects/{project_id}/runs/{run_id}")
	def get_run(project_id: str, run_id: str, since: int = 0, items: bool = True):
		project = project_or_404(project_id)
		run_dir = run_or_404(project, run_id)
		evaluation = read_json(run_dir / "eval.json", None) if items else None
		if evaluation and evaluation.get("items"):
			schema = (read_json(run_dir / "run.json", {}) or {}).get("schema")
			evaluation["labels"] = label_breakdown(evaluation["items"], schema)
			evaluation["refs"] = ref_scores(evaluation["items"], schema, run_resolver(run_dir))
		skipped = read_json(run_dir / "skipped.json", None)
		return {
			**run_summary(run_dir),
			"skipped_examples": {"max_length": skipped["max_length"], "ids": skipped["ids"][:100]} if skipped else None,
			"events": read_events(run_dir, since),
			"log": tail_text(job_paths("train", run_dir)[1]),
			"export_log": tail_text(job_paths("export", run_dir)[1]),
			"predict": {
				"status": read_json(run_dir / "predict" / "status.json", None),
				"result": read_json(run_dir / "predict" / "result.json", None),
				"log": tail_text(job_paths("predict", run_dir)[1]),
			},
			"evaluation": evaluation,
			"files": list_export_files(run_dir),
			"active_job": active_job(),
		}

	@app.delete("/api/projects/{project_id}/runs/{run_id}")
	def delete_run(project_id: str, run_id: str):
		project = project_or_404(project_id)
		run_dir = run_or_404(project, run_id)
		job = jobs.current()
		if job and job.run_dir == run_dir:
			raise HTTPException(409, "Stop this run's job before deleting it.")
		shutil.rmtree(run_dir)
		return {"ok": True}

	@app.post("/api/projects/{project_id}/runs/{run_id}/stop")
	def stop_run(project_id: str, run_id: str):
		run_dir = run_or_404(project_or_404(project_id), run_id)
		return {"stopped": jobs.stop(run_dir)}

	@app.post("/api/projects/{project_id}/runs/{run_id}/resume")
	def resume_run(project_id: str, run_id: str):
		run_dir = run_or_404(project_or_404(project_id), run_id)
		if not run_summary(run_dir)["resumable"]:
			raise HTTPException(409, "Only stopped or failed runs with a saved checkpoint can resume.")
		try:
			jobs.start("train", run_dir, job_env())
		except JobBusy as e:
			raise HTTPException(409, str(e))
		return run_summary(run_dir)

	@app.post("/api/projects/{project_id}/runs/{run_id}/export")
	def export_run(project_id: str, run_id: str, payload: dict = Body(...)):
		run_dir = run_or_404(project_or_404(project_id), run_id)
		if jobs.current():
			raise HTTPException(409, "Another job is running. Wait for it to finish or stop it.")
		hw = hw_cache.get()
		if not hw["can_train"]:
			raise HTTPException(400, "Exporting needs the training libraries (PyTorch, transformers, peft).")
		try:
			prepare_export(run_dir, payload, ws.load_settings(), ws.tools_dir)
			jobs.start("export", run_dir, job_env())
		except RunError as e:
			raise HTTPException(400, str(e))
		except JobBusy as e:
			raise HTTPException(409, str(e))
		return run_summary(run_dir)

	@app.post("/api/projects/{project_id}/runs/{run_id}/predict")
	def predict_run(project_id: str, run_id: str, payload: dict = Body(...)):
		project = project_or_404(project_id)
		run_dir = run_or_404(project, run_id)
		if jobs.current():
			raise HTTPException(409, "Another job is running. Wait for it to finish or stop it.")
		hw = hw_cache.get()
		if not hw["can_train"]:
			raise HTTPException(400, "Trying a model needs the training libraries (PyTorch, transformers, peft).")
		try:
			prepare_prediction(run_dir, payload, project.images_dir)
			jobs.start("predict", run_dir, job_env())
		except RunError as e:
			raise HTTPException(400, str(e))
		except JobBusy as e:
			raise HTTPException(409, str(e))
		return {"ok": True}

	@app.get("/api/projects/{project_id}/runs/{run_id}/snippets")
	def snippets(project_id: str, run_id: str):
		run_dir = run_or_404(project_or_404(project_id), run_id)
		spec = read_json(run_dir / "run.json", {})
		options = read_json(run_dir / "export" / "options.json", {}) or {}
		export_status = read_json(run_dir / "export" / "status.json", {}) or {}
		name = options.get("ollama_name") or model_name(spec)
		manifest = read_json(run_dir / "export" / "llmzempic.json", {}) or {}
		schema, refs = export_contract(spec["schema"], spec.get("datasets"))
		return {
			"name": name,
			"datasets": refs,
			"snippets": all_snippets(
				name, schema, spec["system_prompt"], export_status.get("gguf"), manifest.get("context_size", 4096),
				vision=spec.get("input_kind") == "image", mmproj_file=export_status.get("mmproj"), image_tokens=manifest.get("image_max_tokens"),
				refs=refs,
			),
		}

	@app.post("/api/projects/{project_id}/runs/{run_id}/open")
	def open_folder(project_id: str, run_id: str):
		run_dir = run_or_404(project_or_404(project_id), run_id)
		target = run_dir / "export" if (run_dir / "export").exists() else run_dir
		try:
			if os.name == "nt":
				os.startfile(str(target))  # noqa: S606 - local desktop convenience
			elif sys.platform == "darwin":
				subprocess.Popen(["open", str(target)])
			else:
				subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		except OSError as e:
			raise HTTPException(500, f"Could not open the folder: {e}")
		return {"path": str(target)}

	app.include_router(image_router(project_or_404, run_or_404))
	app.include_router(example_router(ws, project_or_404, checker))
	app.include_router(dataset_router(ws, checker))
	app.include_router(package_router(ws, project_or_404, project_view))
	app.mount("/", WebFiles(directory=str(WEB_DIR), html=True), name="web")
	return app
