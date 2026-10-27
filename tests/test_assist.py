import json
import time

import pytest

from llmzempic import hub_import, synthetic
from test_server import checked, env  # noqa: F401  (fixture)

FEATURES = {"text": {"dtype": "string", "_type": "Value"}, "label": {"names": ["card_lost", "refund", "top_up"], "_type": "ClassLabel"}}
ROWS = [{"text": f"message {i}", "label": i % 3} for i in range(250)] + [{"text": "", "label": 0}]


def fake_fetch(path, params, token=None):
	if path == "info":
		return {"dataset_info": {"default": {"features": FEATURES, "splits": {"train": {"num_examples": len(ROWS)}, "test": {"num_examples": 10}}}}}
	start, length = params["offset"], params["length"]
	return {
		"features": [{"name": k, "type": v} for k, v in FEATURES.items()],
		"rows": [{"row": r} for r in ROWS[start:start + length]],
		"num_rows_total": len(ROWS),
	}


def test_hub_dataset_description_and_import():
	info = hub_import.describe("org/banking", fetch=fake_fetch)["configs"][0]
	assert info["text"] == "text" and info["labels"] == ["label"]
	assert info["splits"][0] == {"name": "train", "rows": 251}
	pages = []
	examples, seen, skipped, features = hub_import.import_examples(
		"org/banking", "default", "train", "text", ["label"], 1000, fetch=fake_fetch, on_page=lambda done, total: pages.append(done))
	assert len(examples) == 250 and skipped == 1 and pages == [100, 200, 251]
	assert examples[1] == {"input": "message 1", "output": {"label": "refund"}}
	schema = hub_import.labels_schema(seen, features)
	assert schema["properties"]["label"]["enum"] == ["card_lost", "refund", "top_up"]
	assert len(hub_import.import_examples("org/banking", "default", "train", "text", ["label"], 30, fetch=fake_fetch)[0]) == 30
	assert hub_import.labels_schema({"ok": [True, False]})["properties"]["ok"] == {"type": "boolean"}
	with pytest.raises(hub_import.HubError):
		hub_import.import_examples("org/banking", "default", "train", "text", [], fetch=fake_fetch)


def test_hub_import_endpoint_sets_the_schema(env, monkeypatch):  # noqa: F811
	client, ws, _ = env
	monkeypatch.setattr(hub_import, "fetch_json", fake_fetch)
	monkeypatch.setattr(hub_import.import_examples, "__defaults__", (5000, None, fake_fetch, None))
	pid = client.post("/api/projects", json={"name": "Banking"}).json()["id"]
	res = client.post(f"/api/projects/{pid}/hub-import", json={
		"dataset": "org/banking", "config": "default", "split": "train", "text": "text", "labels": ["label"], "limit": 60, "set_schema": True,
	})
	assert res.status_code == 200, res.text
	assert res.json()["added"] == 60 and res.json()["view"]["config"]["schema"]["properties"]["label"]["enum"][0] == "card_lost"
	assert checked(client, pid)["counts"]["problems"] == 0
	assert client.post(f"/api/projects/{pid}/hub-import", json={"dataset": "x"}).status_code == 400


SCHEMA = {"type": "object", "properties": {"intent": {"type": "string", "enum": ["billing", "other"]}, "urgent": {"type": "boolean"}}, "required": ["intent", "urgent"]}


def test_synthetic_examples_are_checked_against_the_schema():
	asked = []

	def ask(settings, messages):
		asked.append(messages[1]["content"])
		field, value = ("intent", "billing") if '"intent" must be "billing"' in messages[1]["content"] else (None, None)
		examples = [
			{"input": "Why was I charged twice?", "output": {"intent": "billing", "urgent": False}},
			{"input": "Why was I charged twice?", "output": {"intent": "billing", "urgent": False}},  # duplicate
			{"input": "Refund please", "output": {"intent": "billing"}},  # misses a required field
			{"input": "Hello", "output": {"intent": "other", "urgent": True}},  # not the asked-for value
		]
		return "```json\n" + json.dumps({"examples": examples if field else []}) + "\n```"

	batches = []
	synthetic.generate({}, "Route tickets.", SCHEMA, 5, on_batch=lambda b, done, total: batches.append((b, done, total)), ask=ask)
	assert [(len(b), done, total) for b, done, total in batches] == [(1, 1, 4), (0, 2, 4), (0, 3, 4), (0, 4, 4)]
	assert batches[0][0][0]["id"].startswith("syn-") and batches[0][0][0]["input"] == "Why was I charged twice?"
	assert "hard cases that could be mistaken for \"other\"" in asked[0] and "Route tickets." in asked[0]
	assert synthetic.targets(SCHEMA, ["urgent"]) == [("urgent", True), ("urgent", False)]
	with pytest.raises(synthetic.SynthesisError):
		synthetic.chat({}, [])


def test_synthesis_job_endpoint(env, monkeypatch):  # noqa: F811
	client, ws, _ = env
	pid = client.post("/api/projects", json={"name": "Tickets"}).json()["id"]
	client.put(f"/api/projects/{pid}", json={"config": {"schema": SCHEMA}})
	assert client.post(f"/api/projects/{pid}/synthesize", json={"per_value": 2}).status_code == 400
	client.put("/api/settings", json={"generator_model": "llama3.1"})

	def fake_chat(settings, messages, timeout=300):
		content = messages[1]["content"]
		value = True if '"urgent" must be true' in content else False if '"urgent" must be false' in content else None
		intent = "billing" if '"intent" must be "billing"' in content else "other"
		urgent = value if value is not None else False
		return json.dumps({"examples": [{"input": f"{intent} {urgent} {content[-8:]}", "output": {"intent": intent, "urgent": urgent}}]})

	monkeypatch.setattr(synthetic, "chat", fake_chat)
	monkeypatch.setattr(synthetic.generate, "__defaults__", ((), None, None, lambda: False, fake_chat))
	assert client.post(f"/api/projects/{pid}/synthesize", json={"per_value": 2}).json() == {"ok": True}
	for _ in range(100):
		status = client.get(f"/api/projects/{pid}/synthesize").json()
		if not status["running"]:
			break
		time.sleep(0.05)
	assert status["error"] is None and status["done"] == status["total"] == 4 and status["added"] >= 3
	assert checked(client, pid)["counts"]["all"] == status["added"]


def test_overlap_finds_near_copies_only():
	from llmzempic.overlap import find_overlaps, jaccard, shingles

	held = [
		{"id": "h1", "input": "I was charged twice for my subscription this month"},
		{"id": "h2", "input": "How do I locate my card?"},
		{"id": "h3", "input": "Hello"},
		{"id": "h4", "input": ""},
	]
	train = [
		{"id": "t1", "input": "i was charged TWICE for my subscription this month!"},
		{"id": "t2", "input": "How can I locate my card?"},
		{"id": "t3", "input": "hello"},
		{"id": "t4", "input": "Completely unrelated text about the weather today"},
	]
	result = find_overlaps(held, train)
	assert result["count"] == 2 and result["held_out_ids"] == ["h1", "h3"]
	assert {(p["held_out"], p["train"]) for p in result["pairs"]} == {("h1", "t1"), ("h3", "t3")}
	assert jaccard(shingles("a b c d"), shingles("a b c e")) == 0.5
	# Prefix filtering finds every pair a brute-force comparison finds.
	import random

	rng = random.Random(3)
	words = "card refund charge account login password transfer limit fee app".split()
	texts = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 8))) for _ in range(150)]
	held, train = [{"id": i, "input": t} for i, t in enumerate(texts[:60])], [{"id": i, "input": t} for i, t in enumerate(texts[60:])]
	brute = {h["id"] for h in held for t in train if jaccard(shingles(h["input"]), shingles(t["input"])) >= 0.8}
	assert set(map(int, find_overlaps(held, train)["held_out_ids"])) == brute


def test_eval_rows_from_a_file(tmp_path):
	from argparse import Namespace

	from llmzempic.cli import _eval_rows

	path = tmp_path / "test.jsonl"
	path.write_text('{"input": "a", "output": {"intent": "billing"}}\n{"id": "x", "input": "b", "output": {"intent": "other"}}\n')
	args = Namespace(data=str(path), hub=None, limit=0, label=None, config=None, split="test", text="text")
	assert _eval_rows(args, []) == [{"id": "0", "input": "a", "output": {"intent": "billing"}}, {"id": "x", "input": "b", "output": {"intent": "other"}}]
	args.hub = "org/set"
	with pytest.raises(ValueError):
		_eval_rows(args, [])
