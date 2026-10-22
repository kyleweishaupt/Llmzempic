import json
import time

import pytest
from fastapi.testclient import TestClient

from llmzempic.fileio import read_json, read_jsonl, write_json
from llmzempic.server import create_app
from llmzempic.workspace import Workspace

HARDWARE = {
	"device": "cpu", "gpus": [], "vram_gb": 0, "ram_gb": 16, "can_train": True, "can_4bit": False,
	"can_unsloth": False, "missing_train_packages": [], "ollama": None, "can_train_images": True, "missing_image_packages": [],
}


class StubHardware:
	def get(self, refresh=False):
		return HARDWARE


class StubJobs:
	def __init__(self):
		self.started = []

	def current(self):
		return None

	def start(self, kind, run_dir, env=None):
		self.started.append((kind, run_dir, env))

	def stop(self, run_dir=None):
		return False


@pytest.fixture
def env(tmp_path):
	ws = Workspace(tmp_path / "ws")
	jobs = StubJobs()
	client = TestClient(create_app(ws, jobs, StubHardware()))
	return client, ws, jobs


def checked(client, pid, query=""):
	"""GET the examples once background checks have caught up."""
	for _ in range(200):
		data = client.get(f"/api/projects/{pid}/examples{query}").json()
		if not data["checking"]["running"]:
			return data
		time.sleep(0.05)
	raise AssertionError("checks never finished")


def problems(client, pid):
	return sorted(e["errors"][0] for e in checked(client, pid, "?filter=problems&limit=100")["examples"])


def make_project(client):
	res = client.post("/api/projects", json={"name": "Ticket Router", "template": "classification"})
	assert res.status_code == 200
	return res.json()["id"]


def test_static_ui_served(env):
	client, _, _ = env
	res = client.get("/")
	assert res.status_code == 200
	assert "Llmzempic" in res.text
	assert client.get("/js/main.js").status_code == 200
	assert client.get("/css/tokens.css").status_code == 200
	assert res.headers["cache-control"] == "no-cache"


def test_catalog_includes_fit(env):
	client, _, _ = env
	data = client.get("/api/catalog").json()
	assert data["models"] and all("fit" in m for m in data["models"])
	assert {"quick", "balanced", "thorough"} <= set(data["presets"])


def test_project_flow_to_training_and_export(env):
	client, ws, jobs = env
	pid = make_project(client)
	assert pid == "ticket-router"

	res = client.put(f"/api/projects/{pid}", json={"config": {"instructions": "Route it.", "training": {"epochs": 1}}})
	assert res.status_code == 200
	view = res.json()
	assert view["config"]["training"]["epochs"] == 1
	assert view["config"]["training"]["lora_r"] == 16
	assert view["system_prompt"].startswith("Route it.")

	examples = [{"input": f"text {i}", "output": {"label": "billing" if i % 2 else "technical"}} for i in range(12)]
	examples.append({"input": "bad", "output": {"label": "nope"}})
	client.put(f"/api/projects/{pid}/examples", json={"examples": examples})
	data = checked(client, pid)
	assert data["counts"]["problems"] == 1
	label_counts = {v["value"]: v["count"] for v in data["stats"]["fields"][0]["values"]}
	assert label_counts == {"billing": 6, "technical": 6, "account": 0, "other": 0}
	assert data["stats"]["duplicates"] == {"groups": 0, "conflicts": 0}

	res = client.post(f"/api/projects/{pid}/runs")
	assert res.status_code == 400
	assert "have problems" in res.json()["detail"]

	client.put(f"/api/projects/{pid}/examples", json={"examples": examples[:-1]})
	checked(client, pid)
	res = client.post(f"/api/projects/{pid}/runs")
	assert res.status_code == 200, res.text
	run_id = res.json()["id"]
	assert jobs.started[0][0] == "train"
	run_dir = ws.get(pid).runs_dir / run_id
	spec = read_json(run_dir / "run.json")
	assert spec["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
	assert spec["params_b"] == 1.54
	assert len(read_jsonl(run_dir / "train.jsonl")) + len(read_jsonl(run_dir / "eval.jsonl")) == 12
	assert spec["eval_count"] == 1

	runs = client.get(f"/api/projects/{pid}/runs").json()["runs"]
	assert runs[0]["id"] == run_id

	detail = client.get(f"/api/projects/{pid}/runs/{run_id}").json()
	assert detail["status"]["state"] == "queued"
	assert detail["events"] == []
	assert detail["evaluation"] is None

	items = [{"input": "x", "expected": {"label": "billing"}, "generated": '{"label":"technical"}'}]
	write_json(run_dir / "eval.json", {"count": 1, "exact_match": 0.0, "items": items})
	labels = client.get(f"/api/projects/{pid}/runs/{run_id}").json()["evaluation"]["labels"]
	assert labels[0]["confusions"] == [{"expected": "billing", "predicted": "technical", "unknown": False, "count": 1}]
	(run_dir / "eval.json").unlink()

	res = client.post(f"/api/projects/{pid}/runs/{run_id}/export", json={"gguf": "q8_0"})
	assert res.status_code == 400
	res = client.post(f"/api/projects/{pid}/runs/{run_id}/predict", json={"inputs": ["hello"]})
	assert res.status_code == 400 and "completed" in res.json()["detail"]

	write_json(run_dir / "status.json", {"state": "completed"})
	(run_dir / "adapter").mkdir()

	for bad in ({"inputs": []}, {"inputs": ["  "]}, {"inputs": "hello"}, {"inputs": ["x"] * 11}):
		assert client.post(f"/api/projects/{pid}/runs/{run_id}/predict", json=bad).status_code == 400, bad
	write_json(run_dir / "predict" / "result.json", {"items": ["stale"]})
	res = client.post(f"/api/projects/{pid}/runs/{run_id}/predict", json={"inputs": ["My card was charged twice"]})
	assert res.status_code == 200, res.text
	assert jobs.started[-1][0] == "predict"
	assert read_json(run_dir / "predict" / "request.json") == {"inputs": ["My card was charged twice"]}
	predict = client.get(f"/api/projects/{pid}/runs/{run_id}?items=false").json()["predict"]
	assert predict["status"]["state"] == "queued" and predict["result"] is None
	write_json(run_dir / "predict" / "result.json", {"items": [{"input": "x", "value": {"label": "billing"}}]})
	predict = client.get(f"/api/projects/{pid}/runs/{run_id}").json()["predict"]
	assert predict["result"]["items"][0]["value"] == {"label": "billing"}
	res = client.post(f"/api/projects/{pid}/runs/{run_id}/export", json={"gguf": "q4_k_m", "ollama_name": "Bad Name"})
	assert res.status_code == 400
	res = client.post(f"/api/projects/{pid}/runs/{run_id}/export", json={"gguf": "q4_k_m", "ollama": True})
	assert res.status_code == 200, res.text
	assert jobs.started[-1][0] == "export"
	options = read_json(run_dir / "export" / "options.json")
	assert options["gguf"] == "q4_k_m" and options["ollama_name"] == "ticket-router"

	snippets = client.get(f"/api/projects/{pid}/runs/{run_id}/snippets").json()
	assert set(snippets["snippets"]) == {"php", "csharp", "python", "llama_server"}

	assert client.delete(f"/api/projects/{pid}/runs/{run_id}").status_code == 200
	assert client.get(f"/api/projects/{pid}/runs/{run_id}").status_code == 404


def test_import_and_infer_schema(env):
	client, _, _ = env
	pid = make_project(client)
	csv_text = "input,label\nhello,greeting\nbye,farewell\nhi,greeting\nciao,farewell\n"
	data = client.post(f"/api/projects/{pid}/import?filename=x.csv&mode=replace", content=csv_text.encode()).json()
	assert data["added"] == 4
	assert checked(client, pid)["counts"]["problems"] == 4  # labels don't match the template's enum yet
	schema = client.post(f"/api/projects/{pid}/infer-schema").json()["schema"]
	assert schema["properties"]["label"]["enum"] == ["greeting", "farewell"]
	client.put(f"/api/projects/{pid}", json={"config": {"schema": schema}})
	data = checked(client, pid)
	assert data["counts"]["problems"] == 0
	assert data["stats"]["fields"][0]["values"] == [{"value": "greeting", "count": 2}, {"value": "farewell", "count": 2}]


def test_settings_never_return_token(env):
	client, _, _ = env
	client.put("/api/settings", json={"hf_token": "hf_secret", "llama_cpp_dir": "/opt/llama.cpp"})
	data = client.get("/api/settings").json()
	assert data == {"hf_token_set": True, "llama_cpp_dir": "/opt/llama.cpp"}


def test_unknown_ids_are_rejected(env):
	client, ws, _ = env
	assert client.get("/api/projects/bad%20id").status_code == 404
	assert client.get("/api/projects/nope").status_code == 404
	pid = make_project(client)
	assert client.get(f"/api/projects/{pid}/runs/nope").status_code == 404
	for bad in ("..", "../x", ".hidden", ""):
		with pytest.raises(KeyError):
			ws.get(bad)
		with pytest.raises(KeyError):
			ws.get(pid).run_path(bad)


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # the server only checks the file signature


def test_image_project_flow(env):
	client, ws, jobs = env
	view = client.post("/api/projects", json={"name": "Hot dog", "template": "image-yes-no"}).json()
	pid = view["id"]
	assert view["config"]["input_kind"] == "image"

	assert client.post(f"/api/projects/{pid}/images", content=b"GIF? no").status_code == 400
	names = [client.post(f"/api/projects/{pid}/images", content=PNG + bytes([i])).json()["name"] for i in range(12)]
	assert len(set(names)) == 12
	assert client.post(f"/api/projects/{pid}/images", content=PNG + bytes([0])).json()["name"] == names[0]
	res = client.get(f"/api/projects/{pid}/images/{names[0]}")
	assert res.status_code == 200 and res.headers["content-type"] == "image/png" and res.content == PNG + bytes([0])
	assert client.get(f"/api/projects/{pid}/images/..%2Fproject.json").status_code == 404

	examples = [{"input": "", "image": n, "output": {"hot_dog": i % 2 == 0}} for i, n in enumerate(names)]
	missing = [{"input": "no picture", "output": {"hot_dog": True}}, {"input": "", "image": "0" * 24 + ".png", "output": {"hot_dog": True}}]
	data = client.put(f"/api/projects/{pid}/examples", json={"examples": examples + missing}).json()
	assert problems(client, pid) == ["Image file is missing", "No image"]
	assert data["examples"][0]["image"] == names[0]

	# The UI's JSONL download carries image names; importing it back keeps them.
	jsonl = "\n".join(json.dumps({"input": "", "image": n, "output": {"hot_dog": True}}) for n in names[:2])
	client.post(f"/api/projects/{pid}/import?filename=x.jsonl&mode=replace", content=jsonl.encode())
	data = checked(client, pid)
	assert [e["image"] for e in data["examples"]] == names[:2] and data["counts"]["problems"] == 0

	client.put(f"/api/projects/{pid}/examples", json={"examples": examples})
	client.put(f"/api/projects/{pid}", json={"config": {"base_model": "Qwen/Qwen2.5-1.5B-Instruct"}})
	res = client.post(f"/api/projects/{pid}/runs")
	assert res.status_code == 400 and "can't read images" in res.json()["detail"]

	client.put(f"/api/projects/{pid}", json={"config": {"base_model": "someone/custom-vlm"}})
	checked(client, pid)
	res = client.post(f"/api/projects/{pid}/runs")
	assert res.status_code == 200, res.text
	run_id = res.json()["id"]
	assert res.json()["input_kind"] == "image"
	run_dir = ws.get(pid).runs_dir / run_id
	assert sorted(p.name for p in (run_dir / "images").iterdir()) == sorted(names)
	assert read_json(run_dir / "run.json")["input_kind"] == "image"
	assert all(r["image"] in names for r in read_jsonl(run_dir / "train.jsonl"))
	assert client.get(f"/api/projects/{pid}/runs/{run_id}/images/{names[3]}").status_code == 200

	write_json(run_dir / "status.json", {"state": "completed"})
	(run_dir / "adapter").mkdir()
	predict = f"/api/projects/{pid}/runs/{run_id}/predict"
	assert client.post(predict, json={"inputs": [""]}).status_code == 400
	assert client.post(predict, json={"inputs": [""], "images": ["0" * 24 + ".png"]}).status_code == 400
	fresh = client.post(f"/api/projects/{pid}/images", content=PNG + b"new").json()["name"]
	res = client.post(predict, json={"inputs": [""], "images": [fresh]})
	assert res.status_code == 200, res.text
	assert read_json(run_dir / "predict" / "request.json") == {"inputs": [""], "images": [fresh]}
	assert (run_dir / "images" / fresh).exists()


def test_orphaned_images_are_removed_once_stale(tmp_path):
	import os
	import time

	from llmzempic.images import remove_orphans, save_image

	used, fresh, stale = (save_image(tmp_path, PNG + bytes([i])) for i in range(3))
	old = time.time() - 2 * 24 * 3600
	for name in (used, stale):
		os.utime(tmp_path / name, (old, old))
	remove_orphans(tmp_path, {used})
	assert sorted(p.name for p in tmp_path.iterdir()) == sorted([used, fresh])
