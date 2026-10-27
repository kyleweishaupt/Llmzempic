import json
import math
import random

import pytest
from fastapi.testclient import TestClient

from llmzempic import decisions
from llmzempic.decision_server import create_decision_app
from llmzempic.decision_snippets import all_decision_snippets
from llmzempic.decisions import DecisionError
from llmzempic.exporter import write_deploy_files
from llmzempic.fileio import read_json, read_jsonl
from test_server import checked, env  # noqa: F401  (fixture)

SCHEMA = {
	"type": "object",
	"properties": {
		"intent": {"type": "string", "enum": ["billing", "technical_support", "sales", "other"], "description": "What is the primary intent?"},
		"urgency": {"type": "string", "enum": ["low", "medium", "high", "critical"], "x-score": True},
		"is_toxic": {"type": "boolean", "description": "Is the message toxic?"},
	},
	"required": ["intent", "urgency", "is_toxic"],
}
QUESTIONS, _ = decisions.schema_questions(SCHEMA)


def test_schema_fields_become_typed_questions():
	assert [(q["name"], q["type"]) for q in QUESTIONS] == [("intent", "choice"), ("urgency", "score"), ("is_toxic", "predicate")]
	assert QUESTIONS[2]["options"] == [True, False]
	assert decisions.question_text(QUESTIONS[1]) == "Rate the urgency of the input."
	bad = {"type": "object", "properties": {
		"note": {"type": "string"},
		"maybe": {"type": ["string", "null"], "enum": ["a", "b", None]},
		"one": {"type": "string", "enum": ["a"]},
		"scale": {"type": "string", "enum": [str(i) for i in range(11)], "x-score": True},
	}}
	questions, problems = decisions.schema_questions(bad)
	assert questions == []
	assert len(problems) == 4 and "isn't an option list" in problems[0] and "null" in problems[1]
	assert decisions.schema_questions({"type": "object"})[1]


def test_prompt_numbers_options_in_display_order():
	q = {"name": "pick", "type": "choice", "instructions": "", "options": [f"o{i}" for i in range(12)]}
	assert decisions.option_ids(12)[:2] == ["01", "02"] and decisions.option_ids(9)[-1] == "9"
	order = [11] + list(range(11))
	text = decisions.user_prompt("hello", q, order)
	assert "Input:\nhello\n" in text and "Question: Which pick fits the input?" in text
	assert "\n01. o11\n02. o0\n" in text
	assert decisions.answer_id(q, 11, order) == "01" and decisions.answer_id(q, 0, order) == "02"
	assert "lowest to highest" in decisions.user_prompt("x", QUESTIONS[1])
	assert "1. Yes\n2. No" in decisions.user_prompt("x", QUESTIONS[2])


def test_only_choice_options_are_shuffled():
	rng = random.Random(3)
	shuffled = {tuple(decisions.shuffled_order(QUESTIONS[0], rng)) for _ in range(20)}
	assert len(shuffled) > 1
	assert decisions.shuffled_order(QUESTIONS[1], rng) == [0, 1, 2, 3]
	assert decisions.shuffled_order(QUESTIONS[2], rng) == [0, 1]


def test_gold_items_skip_missing_and_invalid_answers():
	assert decisions.gold_items({"intent": "sales", "is_toxic": False}, QUESTIONS) == [(0, 2), (2, 1)]
	assert decisions.gold_items({"intent": "nope", "is_toxic": 1}, QUESTIONS) == []
	assert decisions.gold_items("text", QUESTIONS) == []


def test_request_fills_trained_questions_and_checks_new_ones():
	state, questions, threshold = decisions.parse_request({"input": "hi"}, QUESTIONS)
	assert state == "hi" and threshold is None
	assert [q["name"] for q in questions] == ["intent", "urgency", "is_toxic"] and all(q["trained"] for q in questions)

	body = {"input": {"ticket": 7}, "min_confidence": 0.6, "questions": [
		{"name": "intent", "choices": ["billing", "other"]},
		{"name": "polite", "type": "noul", "instructions": "Is it polite?"},
		{"name": "stars", "type": "rating", "instructions": "Rate it", "levels": [1, 2, 3]},
	]}
	state, questions, threshold = decisions.parse_request(body, QUESTIONS)
	assert state == '{"ticket":7}' and threshold == 0.6
	assert [(q["type"], q["trained"]) for q in questions] == [("choice", False), ("predicate", False), ("score", False)]
	assert questions[0]["instructions"] == "What is the primary intent?" and questions[2]["options"] == ["1", "2", "3"]

	messages = [{"role": "user", "content": [{"type": "text", "text": "refund please"}]}, {"role": "assistant", "content": "ok"}]
	assert decisions.state_text(messages) == "user: refund please\nassistant: ok"
	for body, message in [
		({}, "input"),
		({"input": "x"}, "questions"),
		({"input": "x", "questions": [{"name": "a"}]}, "type"),
		({"input": "x", "questions": [{"name": "a", "type": "choice", "instructions": "?", "choices": ["x"]}]}, "2 to 255"),
		({"input": "x", "questions": [{"name": "a", "type": "predicate", "instructions": "?"}] * 2}, "different"),
		({"input": "x", "questions": [{"type": "predicate", "instructions": "?"}], "min_confidence": 2}, "min_confidence"),
		({"input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}], "questions": []}, "text only"),
	]:
		with pytest.raises(DecisionError, match=message):
			decisions.parse_request(body)


def test_answers_report_probabilities_confidence_and_refusals():
	probs = [0.7, 0.2, 0.1, 0.0]
	choice = decisions.build_answer(QUESTIONS[0], probs)
	assert choice["choice"] == "billing" and choice["confidence"] == 0.7 and choice["probabilities"]["technical_support"] == 0.2
	score = decisions.build_answer(QUESTIONS[1], [0.1, 0.2, 0.3, 0.4])
	assert score["score"] == pytest.approx(3.0) and score["level"] == "critical"
	predicate = decisions.build_answer(QUESTIONS[2], [0.12, 0.88])
	assert predicate == {"name": "is_toxic", "type": "predicate", "probability": 0.12, "confidence": 0.88}
	refused = decisions.build_answer(QUESTIONS[0], probs, threshold=0.8)
	assert refused["refused"] and refused["choice"] is None and refused["probabilities"]["billing"] == 0.7
	assert decisions.predicted_value(QUESTIONS[2], [0.6, 0.4]) is True
	assert decisions.predicted_value(QUESTIONS[1], [0.1, 0.2, 0.3, 0.4]) == "critical"

	response = decisions.build_response("m", [dict(QUESTIONS[2], trained=False)], [[math.log(0.3), math.log(0.7)]], 42, temperature=5)
	assert response["answers"][0]["probability"] == pytest.approx(0.3)  # untrained questions aren't rescaled
	assert response["usage"] == {"input_tokens": 42, "output_tokens": 0} and response["object"] == "decision"


def test_ece_and_temperature_scaling():
	assert decisions.ece([0.9, 0.9], [True, False]) == pytest.approx(0.4)
	assert decisions.ece([0.75, 0.25], [True, False]) == pytest.approx(0.25)
	rng = random.Random(0)
	samples = []
	for _ in range(400):
		# True probabilities, reported twice as sharp (an overconfident model): the fit should undo it.
		logits = [rng.gauss(0, 1.5) for _ in range(4)]
		probs = decisions.softmax(logits)
		gold = rng.choices(range(4), probs)[0]
		samples.append({"question": 0, "scores": [2 * x for x in logits], "gold": gold})
	assert decisions.fit_temperature(samples) == pytest.approx(2.0, rel=0.2)
	report = decisions.calibration_report(samples, QUESTIONS[:1])
	assert report["calibrated"] and report["after_calibration"]["ece"] < report["ece"]
	assert report["questions"][0]["name"] == "intent" and report["questions"][0]["count"] == 400
	few = decisions.calibration_report(samples[:10], QUESTIONS[:1])
	assert few["temperature"] == 1.0 and not few["calibrated"]
	assert decisions.percentile([5, 1, 3], 0.5) == 3


class FakeReader:
	def __init__(self):
		self.calls = []

	def score(self, state, questions, adapter=True):
		self.calls.append((state, [q["name"] for q in questions]))
		return [[0.0] * (len(q["options"]) - 1) + [1.0] for q in questions], 99


def test_decision_server_answers_in_the_api_shape():
	reader = FakeReader()
	client = TestClient(create_decision_app(reader, "triage", QUESTIONS, temperature=1.0))
	assert client.get("/v1/models").json()["data"][0]["questions"][1]["levels"] == ["low", "medium", "high", "critical"]
	res = client.post("/v1/decisions", json={"model": "triage", "input": "my bill is wrong", "questions": [{"name": "intent"}, {"name": "is_toxic"}]})
	assert res.status_code == 200, res.text
	body = res.json()
	assert [a["name"] for a in body["answers"]] == ["intent", "is_toxic"]
	assert body["answers"][0]["choice"] == "other" and body["answers"][1]["probability"] < 0.5
	assert body["usage"] == {"input_tokens": 99, "output_tokens": 0}
	assert reader.calls == [("my bill is wrong", ["intent", "is_toxic"])]
	bad = client.post("/v1/decisions", json={"input": "x", "questions": [{"name": "new"}]})
	assert bad.status_code == 400 and "type" in bad.json()["error"]["message"]
	assert client.post("/v1/decisions", json=[1]).status_code == 400


def test_snippets_and_export_files_for_decision_runs(tmp_path):
	snippets = all_decision_snippets("ticket triage", QUESTIONS, "/runs/1")
	assert "def decide(" in snippets["python"] and "function ticket_triage_decide(" in snippets["php"]
	assert "class TicketTriageDecisions" in snippets["csharp"] and '--run "/runs/1"' in snippets["curl"]
	compile(snippets["python"], "client.py", "exec")

	spec = {
		"project_name": "Triage", "base_model": "org/model", "answer_mode": "decisions", "questions": QUESTIONS,
		"schema": SCHEMA, "system_prompt": decisions.system_prompt("Triage tickets."),
	}
	metrics = {"decisions": {"temperature": 1.7, "accuracy": 0.9, "after_calibration": {"ece": 0.04}}}
	write_deploy_files(tmp_path, spec, "triage", "triage-q8_0.gguf", 4096, metrics)
	contract = read_json(tmp_path / "decisions.json")
	assert contract["temperature"] == 1.7 and contract["questions"][2] == {"name": "is_toxic", "type": "predicate", "instructions": "Is the message toxic?"}
	assert "{input}" in contract["user_prompts"]["intent"]
	assert read_json(tmp_path / "llmzempic.json")["answer_mode"] == "decisions"
	assert json.loads((tmp_path / "integration" / "decisions-request.json").read_text())["questions"][0]["choices"][0] == "billing"
	assert "llmzempic serve" in (tmp_path / "README.md").read_text() and (tmp_path / "Modelfile").exists()
	assert not (tmp_path / "schema.json").exists()


def test_decision_project_view_and_run(env):  # noqa: F811
	client, ws, jobs = env
	pid = client.post("/api/projects", json={"name": "Triage"}).json()["id"]
	view = client.put(f"/api/projects/{pid}", json={"config": {"answer_mode": "decisions", "instructions": "Triage."}}).json()
	assert view["decision"]["questions"][0]["type"] == "choice" and view["schema_errors"] == []
	assert view["system_prompt"] == decisions.system_prompt("Triage.")

	schema = {"type": "object", "properties": {**SCHEMA["properties"], "note": {"type": "string"}}, "required": ["intent"]}
	view = client.put(f"/api/projects/{pid}", json={"config": {"schema": schema}}).json()
	assert any('"note"' in e for e in view["schema_errors"])
	assert "isn't an option list" in client.post(f"/api/projects/{pid}/runs").json()["detail"]

	client.put(f"/api/projects/{pid}", json={"config": {"schema": SCHEMA}})
	examples = [{"input": f"text {i}", "output": {"intent": "billing" if i % 2 else "sales", "urgency": "low", "is_toxic": i == 3}} for i in range(12)]
	client.put(f"/api/projects/{pid}/examples", json={"examples": examples})
	checked(client, pid)
	res = client.post(f"/api/projects/{pid}/runs")
	assert res.status_code == 200, res.text
	assert res.json()["answer_mode"] == "decisions"
	run_dir = ws.get(pid).runs_dir / res.json()["id"]
	spec = read_json(run_dir / "run.json")
	assert spec["answer_mode"] == "decisions" and [q["name"] for q in spec["questions"]] == ["intent", "urgency", "is_toxic"]
	assert spec["system_prompt"] == decisions.system_prompt("Triage.")
	assert len(read_jsonl(run_dir / "train.jsonl")) + len(read_jsonl(run_dir / "eval.jsonl")) == 12
	snippets = client.get(f"/api/projects/{pid}/runs/{res.json()['id']}/snippets").json()["snippets"]
	assert set(snippets) == {"python", "php", "csharp", "curl"}

	view = client.put(f"/api/projects/{pid}", json={"config": {"input_kind": "image"}}).json()
	assert any("read text" in e for e in view["schema_errors"])


def test_cli_reads_inline_questions_of_any_length(tmp_path):
	from llmzempic.cli import _load_questions

	questions = [{"name": f"q{i}", "type": "predicate", "instructions": "Is it fine? " * 5} for i in range(12)]
	assert _load_questions(json.dumps(questions)) == questions
	path = tmp_path / "questions.json"
	path.write_text(json.dumps({"questions": questions[:1]}))
	assert _load_questions(str(path)) == questions[:1]
