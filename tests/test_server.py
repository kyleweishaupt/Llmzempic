import pytest
from fastapi.testclient import TestClient

from llmzempic.fileio import read_json, read_jsonl, write_json
from llmzempic.server import create_app
from llmzempic.workspace import Workspace

HARDWARE = {
	"device": "cpu", "gpus": [], "vram_gb": 0, "ram_gb": 16, "can_train": True, "can_4bit": False,
	"can_unsloth": False, "missing_train_packages": [], "ollama": None,
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


def make_project(client):
	res = client.post("/api/projects", json={"name": "Ticket Router", "template": "classification"})
	assert res.status_code == 200
	return res.json()["id"]


def test_static_ui_served(env):
	client, _, _ = env
	res = client.get("/")
	assert res.status_code == 200
	assert "Llmzempic" in res.text
	assert client.get("/app.js").status_code == 200


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
	data = client.put(f"/api/projects/{pid}/examples", json={"examples": examples}).json()
	assert len(data["invalid"]) == 1

	res = client.post(f"/api/projects/{pid}/runs")
	assert res.status_code == 400
	assert "don't match" in res.json()["detail"]

	client.put(f"/api/projects/{pid}/examples", json={"examples": examples[:-1]})
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

	res = client.post(f"/api/projects/{pid}/runs/{run_id}/export", json={"gguf": "q8_0"})
	assert res.status_code == 400

	write_json(run_dir / "status.json", {"state": "completed"})
	(run_dir / "adapter").mkdir()
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
	data = client.post(f"/api/projects/{pid}/import", json={"text": csv_text, "filename": "x.csv", "mode": "replace"}).json()
	assert data["imported"] == 4
	assert len(data["invalid"]) == 4  # labels don't match the template's enum yet
	schema = client.post(f"/api/projects/{pid}/infer-schema").json()["schema"]
	assert schema["properties"]["label"]["enum"] == ["greeting", "farewell"]
	client.put(f"/api/projects/{pid}", json={"config": {"schema": schema}})
	assert client.get(f"/api/projects/{pid}/examples").json()["invalid"] == []


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
