import pytest
from fastapi.testclient import TestClient

from llmzempic.server import create_app
from llmzempic.workspace import Workspace
from test_example_api import checked
from test_server import StubHardware, StubJobs

CODES = {"type": "object", "properties": {"b": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}}}, "required": ["b"]}


@pytest.fixture
def client(tmp_path):
	return TestClient(create_app(Workspace(tmp_path / "ws"), StubJobs(), StubHardware()))


def upload(client, text, name="Codes", filename="codes.txt", **params):
	query = "&".join(f"{k}={v}" for k, v in {"name": name, "filename": filename, **params}.items())
	return client.post(f"/api/datasets?{query}", content=text.encode("utf-8"))


def test_upload_list_search(client):
	meta = upload(client, "V1  First\nV2  Second\nX9  Other\n").json()
	assert (meta["id"], meta["name"], meta["rows"]) == ("codes", "Codes", 3)
	listed = client.get("/api/datasets").json()["datasets"]
	assert [(d["id"], d["used_by"]) for d in listed] == [("codes", [])]
	rows = client.get("/api/datasets/codes/search?q=V").json()["rows"]
	assert [r["value"] for r in rows] == ["V1", "V2"]
	assert client.get("/api/datasets/nope/search?q=V").status_code == 404


def test_preview_and_column_choice(client):
	info = client.post("/api/datasets/preview", json={"text": "code,label,notes\nA,Alpha,x\n", "filename": "x.csv"}).json()
	assert (info["value_column"], info["description_column"]) == ("code", "label")
	meta = upload(client, "code,label,notes\nA,Alpha,x\nB,Beta,y\n", filename="x.csv", value_column="code", description_column="notes").json()
	assert meta["rows"] == 2
	assert client.get(f"/api/datasets/{meta['id']}/search?q=A").json()["rows"][0] == {"value": "A", "description": "x"}


def test_delete_refused_while_used(client):
	upload(client, "V1\n")
	pid = client.post("/api/projects", json={"name": "Coder"}).json()["id"]
	client.put(f"/api/projects/{pid}", json={"config": {"schema": CODES}})
	assert client.get("/api/datasets").json()["datasets"][0]["used_by"] == ["Coder"]
	res = client.delete("/api/datasets/codes")
	assert res.status_code == 409 and res.json()["detail"] == "Used by: Coder. Remove it from their schemas first."
	client.put(f"/api/projects/{pid}", json={"config": {"schema": {"type": "object"}}})
	assert client.delete("/api/datasets/codes").status_code == 200
	assert client.get("/api/datasets").json()["datasets"] == []


def test_replace_rechecks_referencing_project(client):
	upload(client, "V1\n")
	pid = client.post("/api/projects", json={"name": "Coder"}).json()["id"]
	client.put(f"/api/projects/{pid}", json={"config": {"schema": CODES}})
	client.put(f"/api/projects/{pid}/examples", json={"examples": [{"input": "a", "output": {"b": ["V3"]}}]})
	assert checked(client, pid)["counts"]["problems"] == 1
	meta = client.put("/api/datasets/codes?filename=v.txt", content=b"V1\nV3\n").json()
	assert meta["rows"] == 2
	assert checked(client, pid)["counts"]["problems"] == 0


def test_bad_file_is_a_400(client):
	res = upload(client, "\n \n")
	assert res.status_code == 400 and res.json()["detail"] == "No values found in the file."
	res = client.post("/api/datasets/preview", json={"text": "", "filename": "x.csv"})
	assert res.status_code == 200
