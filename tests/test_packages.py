import json
import zipfile

import pytest

from llmzempic.fileio import read_json, write_json
from llmzempic.images import save_image
from llmzempic.packages import PackageError, duplicate_project, export_package, import_package
from llmzempic.workspace import Workspace

CODES = {"type": "object", "properties": {"b": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}}}}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def dataset(ws, tmp_path, text, name="Codes"):
	source = tmp_path / f"{name}-{abs(hash(text))}.txt"
	source.write_text(text, encoding="utf-8")
	return ws.library.create(name, source, source.name)


def make_project(ws, tmp_path):
	dataset(ws, tmp_path, "V1\nV2\n")
	project = ws.create("Coder")
	project.save_config({"schema": CODES, "instructions": "Code it."})
	image = save_image(project.images_dir, PNG)
	project.examples().import_records([
		{"id": "a", "input": "first", "output": {"b": ["V1"]}},
		{"id": "b", "input": "", "image": image, "output": {"b": ["V2"]}},
	], "append")
	run = project.runs_dir / "20260101-000000"
	(run / "adapter").mkdir(parents=True)
	(run / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
	(run / "checkpoints" / "checkpoint-5").mkdir(parents=True)
	write_json(run / "run.json", {"project_id": project.id, "project_name": "Coder"})
	write_json(run / "status.json", {"state": "completed"})
	return project


def rows(project):
	return [{k: r.get(k) for k in ("input", "image", "output")} for r in project.examples().iter_rows()]


def test_round_trip_into_fresh_workspace(tmp_path):
	source_ws, target_ws = Workspace(tmp_path / "a"), Workspace(tmp_path / "b")
	project = make_project(source_ws, tmp_path)
	dataset(target_ws, tmp_path, "OTHER\n")  # takes the id "codes" in the target
	package = export_package(source_ws, project, tmp_path / "coder.llmz", include_run=True)
	imported = import_package(target_ws, package)
	config = imported.load_config()
	assert config["name"] == "Coder" and config["instructions"] == "Code it."
	assert config["schema"]["properties"]["b"]["items"]["x-dataset"] == "codes-2"
	assert target_ws.library.get("codes-2").missing(["V1", "V2"]) == set()
	assert rows(imported) == rows(project)
	assert (imported.images_dir / rows(imported)[1]["image"]).read_bytes() == PNG
	run = imported.runs_dir / "20260101-000000"
	assert (run / "adapter" / "adapter_model.safetensors").read_bytes() == b"weights"
	assert not (run / "checkpoints").exists()
	assert read_json(run / "run.json")["project_id"] == imported.id


def test_runs_left_out_by_default(tmp_path):
	source_ws, target_ws = Workspace(tmp_path / "a"), Workspace(tmp_path / "b")
	imported = import_package(target_ws, export_package(source_ws, make_project(source_ws, tmp_path), tmp_path / "p.llmz"))
	assert imported.run_ids() == []


def test_dataset_reused_by_hash(tmp_path):
	source_ws, target_ws = Workspace(tmp_path / "a"), Workspace(tmp_path / "b")
	project = make_project(source_ws, tmp_path)
	dataset(target_ws, tmp_path, "V2\nV1\n", name="Mine")
	imported = import_package(target_ws, export_package(source_ws, project, tmp_path / "p.llmz"))
	assert imported.load_config()["schema"]["properties"]["b"]["items"]["x-dataset"] == "mine"
	assert [m["id"] for m in target_ws.library.list()] == ["mine"]


def test_duplicate_keeps_original_untouched(tmp_path):
	ws = Workspace(tmp_path / "a")
	project = make_project(ws, tmp_path)
	copy = duplicate_project(ws, project)
	assert copy.id != project.id and copy.load_config()["name"] == "Coder copy"
	assert rows(copy) == rows(project) and [m["id"] for m in ws.library.list()] == ["codes"]
	copy.examples().update("a", {"input": "changed"})
	assert project.examples().get("a")["input"] == "first"


def test_zip_slip_rejected(tmp_path):
	evil = tmp_path / "evil.llmz"
	with zipfile.ZipFile(evil, "w") as z:
		z.writestr("llmzempic-package.json", json.dumps({"format": 1, "name": "x", "datasets": [], "run": None}))
		z.writestr("project/project.json", "{}")
		z.writestr("../escaped.txt", "nope")
	ws = Workspace(tmp_path / "ws")
	with pytest.raises(PackageError):
		import_package(ws, evil)
	assert not (tmp_path / "escaped.txt").exists() and ws.projects() == []
	with pytest.raises(PackageError):
		import_package(ws, _not_a_zip(tmp_path))



def evil_package(tmp_path, manifest, members):
	path = tmp_path / "evil.llmz"
	with zipfile.ZipFile(path, "w") as z:
		z.writestr("llmzempic-package.json", json.dumps({"format": 1, "name": "x", "datasets": [], "run": None, **manifest}))
		z.writestr("project/project.json", "{}")
		for name, text in members.items():
			z.writestr(name, text)
	return path


def test_double_slash_member_cannot_escape(tmp_path):
	# "//" collapses away in the path check, but after the prefix is cut the rest is an absolute path.
	outside = tmp_path / "outside" / "pwned.txt"
	target = outside.relative_to(outside.anchor).as_posix()  # no drive letter: "C:" is rejected anyway
	ws = Workspace(tmp_path / "ws")
	with pytest.raises(PackageError):
		import_package(ws, evil_package(tmp_path, {}, {f"project/images//{target}": "nope"}))
	assert not (tmp_path / "outside").exists()


def test_manifest_run_and_data_set_ids_must_be_plain_names(tmp_path):
	ws = Workspace(tmp_path / "ws")
	with pytest.raises(PackageError):
		import_package(ws, evil_package(tmp_path, {"run": "../../run-escape"}, {"run/../../run-escape/run.json": "{}"}))
	with pytest.raises(PackageError):
		import_package(ws, evil_package(tmp_path, {"datasets": [{"id": "../x", "hash": "h", "name": "x"}]}, {}))
	assert not (tmp_path / "run-escape").exists() and ws.projects() == []

def _not_a_zip(tmp_path):
	path = tmp_path / "plain.llmz"
	path.write_text("hello", encoding="utf-8")
	return path


def test_package_routes_round_trip(tmp_path):
	from fastapi.testclient import TestClient

	from llmzempic.server import create_app
	from test_server import StubHardware, StubJobs

	ws = Workspace(tmp_path / "ws")
	project = make_project(ws, tmp_path)
	client = TestClient(create_app(ws, StubJobs(), StubHardware()))
	# Building a large package takes a while, so it's prepared first and then downloaded.
	ready = client.post(f"/api/projects/{project.id}/package", json={"include_run": True}).json()
	assert ready["filename"] == "coder.llmz" and ready["size"] > 0
	res = client.get(f"/api/packages/{ready['token']}")
	assert res.status_code == 200 and res.headers["content-disposition"].endswith('coder.llmz"')
	assert not list(ws.root.glob(".package-*"))  # the temp file is removed after sending
	assert client.get(f"/api/packages/{ready['token']}").status_code == 404
	assert client.get("/api/packages/not-a-token").status_code == 404
	view = client.post("/api/packages", content=res.content).json()
	assert view["id"] == "coder-2" and view["config"]["name"] == "Coder"
	assert client.post("/api/packages", content=b"not a zip").status_code == 400
	assert client.post(f"/api/projects/{project.id}/duplicate").json()["config"]["name"] == "Coder copy"


def test_export_includes_only_uploaded_images(tmp_path):
	# Rows imported before image names were checked can name any file on disk.
	ws = Workspace(tmp_path / "ws")
	project = make_project(ws, tmp_path)
	secret = tmp_path / "private-notes.txt"
	secret.write_text("private", encoding="utf-8")
	with project.examples().connect() as db:
		db.execute("UPDATE examples SET image = ? WHERE id = 'a'", (str(secret),))
		db.commit()
	export_package(ws, project, tmp_path / "out.llmz")
	with zipfile.ZipFile(tmp_path / "out.llmz") as z:
		assert not any("private-notes" in name for name in z.namelist())
		assert sum(name.startswith("project/images/") for name in z.namelist()) == 1
