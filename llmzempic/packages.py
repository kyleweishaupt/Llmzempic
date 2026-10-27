"""Project packages: one .llmz zip with a project's settings, examples, images and the data sets its
schema uses (optionally the latest trained run), to move work to another machine or start a copy."""

import io
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from .fileio import read_json, write_json
from .images import IMAGE_NAME
from .refs import referenced_ids, rewrite_ids
from .runs import run_summary
from .workspace import SAFE_ID

FORMAT = 1
MANIFEST = "llmzempic-package.json"
# serve/ holds production decision logs (with their inputs): they stay on the machine that served them.
RUN_SKIP = ("checkpoints", "export", "predict", "serve")


class PackageError(Exception):
	pass


def _latest_completed_run(project):
	for run_id in project.run_ids():
		run_dir = project.runs_dir / run_id
		if run_summary(run_dir)["status"].get("state") == "completed" and (run_dir / "adapter").is_dir():
			return run_dir
	return None


def export_package(workspace, project, dest, include_run=False):
	"""Write the package to dest (streamed; a 200k-example project never sits in memory)."""
	config = project.load_config()
	store = project.examples()
	datasets = [workspace.library.get(i) for i in sorted(referenced_ids(config["schema"])) if i in {m["id"] for m in workspace.library.list()}]
	run_dir = _latest_completed_run(project) if include_run else None
	manifest = {
		"format": FORMAT,
		"name": config.get("name") or project.id,
		"datasets": [{k: d.meta.get(k) for k in ("id", "hash", "name")} for d in datasets],
		"run": run_dir.name if run_dir else None,
	}
	dest = Path(dest)
	with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
		z.writestr(MANIFEST, json.dumps(manifest, indent=1))
		z.writestr("project/project.json", json.dumps(config, ensure_ascii=False, indent=1))
		with z.open("project/examples.jsonl", "w", force_zip64=True) as f:
			for row in store.iter_rows():
				row.pop("errors", None)
				f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
		for name in sorted(store.referenced_images()):
			if IMAGE_NAME.match(name) and (project.images_dir / name).is_file():
				z.write(project.images_dir / name, f"project/images/{name}")
		for d in datasets:
			z.write(d.db_file, f"datasets/{d.meta['id']}/{d.db_file.name}")
			z.write(d.path / "meta.json", f"datasets/{d.meta['id']}/meta.json")
		if run_dir:
			for path in sorted(run_dir.rglob("*")):
				rel = path.relative_to(run_dir)
				if path.is_file() and rel.parts[0] not in RUN_SKIP:
					z.write(path, f"run/{run_dir.name}/{rel.as_posix()}")
	return dest


def _safe_names(z):
	for name in z.namelist():
		segments = name.rstrip("/").split("/")
		if name.startswith(("/", "\\")) or ".." in segments or "" in segments or ":" in name or "\\" in name:
			raise PackageError(f"The package contains an unsafe path: {name}")
	return set(z.namelist())


def _extract(z, prefix, dest):
	root = Path(dest).resolve()
	for name in z.namelist():
		if name.startswith(prefix) and not name.endswith("/"):
			target = (root / PurePosixPath(name[len(prefix):])).resolve()
			if not target.is_relative_to(root):
				raise PackageError(f"The package contains an unsafe path: {name}")
			target.parent.mkdir(parents=True, exist_ok=True)
			with z.open(name) as src, open(target, "wb") as out:
				shutil.copyfileobj(src, out)


def import_package(workspace, src):
	"""Create a new project from a package. Data sets already in the library (same content) are reused."""
	try:
		z = zipfile.ZipFile(src)
	except zipfile.BadZipFile:
		raise PackageError("This isn't a Llmzempic package.")
	with z:
		names = _safe_names(z)
		if MANIFEST not in names or "project/project.json" not in names:
			raise PackageError("This isn't a Llmzempic package.")
		manifest = json.loads(z.read(MANIFEST))
		if manifest.get("format") != FORMAT:
			raise PackageError("This package was made by a newer version of Llmzempic.")
		ids = [d.get("id") for d in manifest.get("datasets") or []] + ([manifest["run"]] if manifest.get("run") else [])
		if not all(isinstance(i, str) and SAFE_ID.match(i) for i in ids):
			raise PackageError("The package names a data set or run with an unsafe id.")
		config = json.loads(z.read("project/project.json"))
		mapping = {}
		for d in manifest.get("datasets") or []:
			existing = workspace.library.by_hash(d.get("hash"))
			if existing:
				mapping[d["id"]] = existing.meta["id"]
				continue
			staging = Path(tempfile.mkdtemp(prefix=".package-", dir=workspace.root))
			_extract(z, f"datasets/{d['id']}/", staging)
			mapping[d["id"]] = workspace.library.adopt(staging, d.get("name") or d["id"])["id"]
		config["schema"] = rewrite_ids(config.get("schema") or {}, mapping)
		project = workspace.create(config.get("name") or manifest.get("name") or "Imported")
		project.save_config({k: v for k, v in config.items() if k not in ("created", "updated")})
		_extract(z, "project/images/", project.images_dir)
		if "project/examples.jsonl" in names:
			with io.TextIOWrapper(z.open("project/examples.jsonl"), encoding="utf-8") as f:
				project.examples().import_records((json.loads(line) for line in f if line.strip()), "append")
		if manifest.get("run"):
			run_dir = project.runs_dir / manifest["run"]
			_extract(z, f"run/{manifest['run']}/", run_dir)
			spec = read_json(run_dir / "run.json", {}) or {}
			write_json(run_dir / "run.json", {**spec, "project_id": project.id})
	return project


def duplicate_project(workspace, project):
	"""A copy with every example, image and setting; the copy shares the original's data sets."""
	temp = workspace.root / f".duplicate-{uuid.uuid4().hex}.llmz"
	try:
		export_package(workspace, project, temp)
		copy = import_package(workspace, temp)
	finally:
		temp.unlink(missing_ok=True)
	copy.save_config({"name": f"{project.load_config().get('name') or project.id} copy"})
	return copy
