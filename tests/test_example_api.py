import json
import time

import pytest
from fastapi.testclient import TestClient

from llmzempic.server import create_app
from llmzempic.workspace import Workspace
from test_server import StubHardware, StubJobs

LABELS = {"type": "object", "properties": {"label": {"type": "string", "enum": ["a", "b"]}}, "required": ["label"]}


@pytest.fixture
def client(tmp_path):
	return TestClient(create_app(Workspace(tmp_path / "ws"), StubJobs(), StubHardware()))


def checked(client, pid, query=""):
	"""GET the examples once background checks have caught up."""
	for _ in range(200):
		data = client.get(f"/api/projects/{pid}/examples{query}").json()
		if not data["checking"]["running"]:
			return data
		time.sleep(0.05)
	raise AssertionError("checks never finished")


def project(client, n=0, schema=LABELS):
	pid = client.post("/api/projects", json={"name": "Api"}).json()["id"]
	client.put(f"/api/projects/{pid}", json={"config": {"schema": schema}})
	if n:
		rows = [{"id": f"e{i}", "input": f"text {i}", "output": {"label": "ab"[i % 2]}} for i in range(n)]
		client.put(f"/api/projects/{pid}/examples", json={"examples": rows})
	return pid


def test_paging_and_filters_over_api(client):
	pid = project(client, 45)
	data = checked(client, pid, "?limit=20")
	assert [e["id"] for e in data["examples"]][:2] == ["e0", "e1"] and len(data["examples"]) == 20
	assert data["total"] == 45 and data["counts"] == {"all": 45, "problems": 0, "duplicates": 0}
	page3 = client.get(f"/api/projects/{pid}/examples?offset=40&limit=20&stats=0").json()
	assert [e["id"] for e in page3["examples"]] == ["e40", "e41", "e42", "e43", "e44"] and "stats" not in page3
	only_b = client.get(f"/api/projects/{pid}/examples?field=label&value=%22b%22&kind=enum&limit=100").json()
	assert only_b["total"] == 22 and all(e["output"]["label"] == "b" for e in only_b["examples"])
	assert client.get(f"/api/projects/{pid}/examples?search=text%2044").json()["total"] == 1


def test_edits_are_checked_immediately(client):
	pid = project(client, 2)
	checked(client, pid)
	row = client.put(f"/api/projects/{pid}/examples/e1", json={"output": {"label": "zzz"}}).json()
	assert row["errors"] and "zzz" in row["errors"][0]
	added = client.post(f"/api/projects/{pid}/examples", json={"examples": [{"input": "", "output": {"label": "a"}}]}).json()["examples"]
	assert added[0]["errors"] == ["Input is empty"]
	assert client.get(f"/api/projects/{pid}/examples").json()["counts"]["problems"] == 2
	assert client.put(f"/api/projects/{pid}/examples/nope", json={"input": "x"}).status_code == 404


def test_bulk_delete_and_undo_over_api(client):
	pid = project(client, 10)
	res = client.post(f"/api/projects/{pid}/examples/bulk", json={"op": "delete", "filter": {"label": {"field": "label", "value": "a", "kind": "enum"}}}).json()
	assert res["affected"] == 5 and checked(client, pid)["total"] == 5
	assert client.post(f"/api/projects/{pid}/examples/undo", json={"op": res["undo"]}).json() == {"restored": 5}
	res = client.post(f"/api/projects/{pid}/examples/bulk", json={"op": "set_output", "ids": ["e0", "e1"], "output": {"label": "b"}}).json()
	assert res["affected"] == 2
	assert checked(client, pid, "?field=label&value=%22b%22&kind=enum")["total"] == 6
	client.post(f"/api/projects/{pid}/examples/undo", json={"op": res["undo"]})
	assert checked(client, pid, "?field=label&value=%22b%22&kind=enum")["total"] == 5
	assert client.post(f"/api/projects/{pid}/examples/bulk", json={"op": "explode", "ids": []}).status_code == 400


def test_streaming_import_reports_bad_lines(client):
	pid = project(client)
	lines = [json.dumps({"input": f"t{i}", "output": {"label": "a"}}) for i in range(4996)]
	lines[10:10] = ["", "   ", "{not json", '{"input": "no output"}', "[1, 2", "nope"]
	body = "﻿" + "\r\n".join(lines) + "\r\n"
	res = client.post(f"/api/projects/{pid}/import?filename=big.jsonl&mode=append", content=body.encode("utf-8")).json()
	assert res["added"] == 4996 and res["updated"] == 0
	assert res["warning_count"] == 4
	assert res["warnings"][0].startswith("Line 13:") and any(w.startswith("Line 14:") for w in res["warnings"])
	assert checked(client, pid)["counts"] == {"all": 4996, "problems": 0, "duplicates": 0}


def test_import_with_nothing_usable_is_an_error(client):
	pid = project(client)
	res = client.post(f"/api/projects/{pid}/import?filename=x.jsonl&mode=append", content=b"{oops\n")
	assert res.status_code == 400 and "Line 1" in res.json()["detail"]


def test_import_update_by_id(client):
	pid = project(client, 3)
	body = "\n".join(json.dumps(r) for r in [
		{"id": "e1", "input": "changed", "output": {"label": "b"}},
		{"id": "new", "input": "fresh", "output": {"label": "a"}},
	])
	res = client.post(f"/api/projects/{pid}/import?filename=x.jsonl&mode=update", content=body.encode()).json()
	assert (res["added"], res["updated"]) == (1, 1)
	rows = {e["id"]: e for e in checked(client, pid)["examples"]}
	assert rows["e1"]["input"] == "changed" and "new" in rows and len(rows) == 4


def test_export_streams_jsonl(client):
	pid = project(client, 3)
	res = client.get(f"/api/projects/{pid}/examples/export")
	assert res.status_code == 200 and "attachment" in res.headers["content-disposition"]
	rows = [json.loads(line) for line in res.text.splitlines()]
	assert [r["id"] for r in rows] == ["e0", "e1", "e2"] and set(rows[0]) == {"id", "input", "output"}


def test_unknown_dataset_reported_in_schema_errors(client):
	schema = {"type": "object", "properties": {"c": {"type": "string", "x-dataset": "gone"}}}
	pid = project(client, schema=schema)
	assert client.get(f"/api/projects/{pid}").json()["schema_errors"] == ['Data set "gone" no longer exists.']


def test_other_sites_cannot_change_data(client):
	# The server has no login; a page on another site could otherwise post a file as a "simple" request.
	pid = project(client, 5)
	body = b'{"input": "from elsewhere", "output": {"label": "a"}}\n'
	url = f"/api/projects/{pid}/import?filename=x.jsonl&mode=replace"
	assert client.post(url, content=body, headers={"Content-Type": "text/plain", "Origin": "https://evil.example"}).status_code == 403
	assert client.post(url, content=body, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
	assert checked(client, pid)["total"] == 5
	assert client.post(url, content=body, headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}).status_code == 200
