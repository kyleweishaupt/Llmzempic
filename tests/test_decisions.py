import json
import math
import random

import pytest
from fastapi.testclient import TestClient

from llmzempic import decisions
from llmzempic.decision_engine import DecisionModel
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
	state, questions, settings = decisions.parse_request({"input": "hi"}, QUESTIONS)
	assert state == "hi" and settings == {"min_confidence": None, "target_accuracy": 0.9, "samples": 1, "image": None}
	assert [q["name"] for q in questions] == ["intent", "urgency", "is_toxic"] and all(q["trained"] for q in questions)

	body = {"input": {"ticket": 7}, "min_confidence": 0.6, "questions": [
		{"name": "intent", "choices": ["billing", "other"]},
		{"name": "polite", "type": "noul", "instructions": "Is it polite?"},
		{"name": "stars", "type": "rating", "instructions": "Rate it", "levels": [1, 2, 3]},
	]}
	state, questions, settings = decisions.parse_request(body, QUESTIONS)
	assert state == '{"ticket":7}' and settings["min_confidence"] == 0.6
	assert [(q["type"], q["trained"]) for q in questions] == [("choice", True), ("predicate", False), ("score", False)]
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
		({"input": "x", "questions": [{"type": "predicate", "instructions": "?"}], "samples": 9}, "samples"),
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
	usage = {"input_tokens": 300, "cached_tokens": 200}
	assert decisions.build_response("m", [], [], usage)["usage"]["input_tokens_details"] == {"cached_tokens": 200}


def test_auto_thresholds_follow_the_held_out_precision_curve():
	confidences = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]
	correct = [True, True, True, True, False, True, False, False]
	curve = decisions.precision_curve(confidences, correct)
	assert curve[0] == [0.95, 1.0, 0.125] and curve[-1] == [0.4, 0.625, 1.0]
	assert decisions.auto_threshold(curve, 0.9) == 0.8
	assert decisions.auto_threshold(curve, 0.8) == 0.6
	assert decisions.auto_threshold([[0.5, 0.4, 1.0]], 0.9) == 1.0
	settings = decisions.parse_settings({"target_accuracy": 0.8})
	assert settings["min_confidence"] == "auto"
	assert decisions.question_threshold(QUESTIONS[0], settings, {"intent": curve}) == 0.6
	assert decisions.question_threshold(dict(QUESTIONS[0], trained=False), settings, {"intent": curve}) is None
	answer = decisions.build_response("m", QUESTIONS[:1], [[0.0, -2.0, -2.0, -2.0]], 1, settings={"min_confidence": "auto"}, curves={"intent": curve})["answers"][0]
	assert answer["refused"] and answer["choice"] is None


def test_training_orders_shuffle_and_sometimes_cut_options():
	rng = random.Random(1)
	q = {"name": "pick", "type": "choice", "options": list("abcdef")}
	orders = [decisions.shuffled_order(q, rng, 2, subset_share=0.5) for _ in range(200)]
	assert all(2 in o and len(set(o)) == len(o) for o in orders)
	sizes = {len(o) for o in orders}
	assert 6 in sizes and min(sizes) == 2
	assert decisions.user_prompt("x", q, [4, 2]).endswith("Options:\n1. e\n2. c\n\nAnswer with the number of one option.")
	assert decisions.answer_id(q, 2, [4, 2]) == "2"


def test_samples_read_choice_questions_in_fixed_extra_orders():
	orders = decisions.sample_orders(QUESTIONS[0], 3)
	assert orders[0] == [0, 1, 2, 3] and len({tuple(o) for o in orders}) == 3
	assert orders == decisions.sample_orders(QUESTIONS[0], 3)
	assert decisions.sample_orders(QUESTIONS[1], 3) == [[0, 1, 2, 3]]
	combined = decisions.combine_samples([[math.log(0.8), math.log(0.2)], [math.log(0.4), math.log(0.6)]])
	assert decisions.softmax(combined) == pytest.approx([0.6, 0.4])


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

	vision = False

	def score(self, state, questions, adapter=True, samples=1, image=None):
		self.calls.append((state, [q["name"] for q in questions], samples))
		return [[0.0] * (len(q["options"]) - 1) + [1.0] for q in questions], {"input_tokens": 99, "cached_tokens": 0}


def test_decision_server_answers_in_the_api_shape():
	reader = FakeReader()
	client = TestClient(create_decision_app(DecisionModel(reader, "triage", QUESTIONS)))
	assert client.get("/v1/models").json()["data"][0]["questions"][1]["levels"] == ["low", "medium", "high", "critical"]
	res = client.post("/v1/decisions", json={"model": "triage", "input": "my bill is wrong", "questions": [{"name": "intent"}, {"name": "is_toxic"}]})
	assert res.status_code == 200, res.text
	body = res.json()
	assert [a["name"] for a in body["answers"]] == ["intent", "is_toxic"]
	assert body["answers"][0]["choice"] == "other" and body["answers"][1]["probability"] < 0.5
	assert body["usage"] == {"input_tokens": 99, "output_tokens": 0}
	assert reader.calls == [("my bill is wrong", ["intent", "is_toxic"], 1)]
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
	template = {"before": "<|user|>", "after": "<|end|><|assistant|>", "answer_prefix": " "}
	curve = [[0.9, 1.0, 0.5], [0.5, 0.8, 1.0]]
	write_deploy_files(tmp_path, spec, "triage", "triage-q8_0.gguf", 4096, metrics, template=template, curves={"intent": curve})
	contract = read_json(tmp_path / "decisions.json")
	toxic = contract["questions"][2]
	assert contract["temperature"] == 1.7 and toxic["instructions"] == "Is the message toxic?" and toxic["numbers"] == ["1", "2"]
	assert contract["questions"][0]["prompt"] == decisions.user_prompt("{input}", QUESTIONS[0])
	assert contract["questions"][0]["auto_min_confidence"] == 0.9 and "auto_min_confidence" not in toxic
	restored = [decisions.from_api_question(q) for q in contract["questions"]]
	assert [decisions.user_prompt("x", q) for q in restored] == [decisions.user_prompt("x", q) for q in QUESTIONS]
	manifest = read_json(tmp_path / "llmzempic.json")
	assert manifest["answer_mode"] == "decisions" and manifest["adapter"] == "adapter"
	assert json.loads((tmp_path / "integration" / "decisions-request.json").read_text())["questions"][0]["choices"][0] == "billing"
	assert "llmzempic serve" in (tmp_path / "README.md").read_text() and (tmp_path / "Modelfile").exists()
	assert not (tmp_path / "schema.json").exists()

	# The vLLM client scores each option number as part of the prompt (echo + logprobs).
	namespace = {}
	exec(compile((tmp_path / "integration" / "triage_vllm.py").read_text(), "triage_vllm.py", "exec"), namespace)
	requests = []

	def fake_post(url, body):
		requests.append((url, body))
		choices = []
		for i, prompt in enumerate(body["prompt"]):
			number = prompt[-1]
			# One token per character; only the option number's tokens count.
			logprobs = [None] + [-5.0] * (len(prompt) - 2) + [-0.2 * int(number)]
			choices.append({"index": i, "logprobs": {"token_logprobs": logprobs + [-1.0], "text_offset": list(range(len(prompt) + 1))}})
		return {"choices": choices}

	namespace["_post"] = fake_post
	answers = namespace["decide"]("refund me", names=["intent", "is_toxic"], min_confidence="auto")
	assert requests[0][0] == "http://localhost:8000/v1/completions" and requests[0][1]["echo"] is True
	assert requests[0][1]["prompt"][1].startswith("<|user|>Input:\nrefund me\n") and requests[0][1]["prompt"][1].endswith("<|end|><|assistant|> 2")
	want = decisions.softmax([-0.2 * k / 1.7 for k in range(1, 5)])
	assert answers["intent"]["probabilities"]["billing"] == pytest.approx(want[0], abs=1e-4)
	assert answers["intent"]["refused"] and answers["intent"]["choice"] is None  # below its 0.9 auto threshold
	assert answers["is_toxic"]["probability"] > 0.5 and "refused" not in answers["is_toxic"]


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
	assert view["schema_errors"] == [] and view["decision"]["questions"]


def test_cli_reads_inline_questions_of_any_length(tmp_path):
	from llmzempic.cli import _load_questions

	questions = [{"name": f"q{i}", "type": "predicate", "instructions": "Is it fine? " * 5} for i in range(12)]
	assert _load_questions(json.dumps(questions)) == questions
	path = tmp_path / "questions.json"
	path.write_text(json.dumps({"questions": questions[:1]}))
	assert _load_questions(str(path)) == questions[:1]


def test_inputs_can_carry_one_image():
	import base64

	png = base64.b64encode(b"\x89PNG fake").decode()
	messages = [{"role": "user", "content": [{"type": "text", "text": "Is this a hot dog?"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}]
	assert decisions.parse_input(messages) == ("user: Is this a hot dog?", b"\x89PNG fake")
	assert decisions.parse_input([{"role": "user", "content": "", "images": [png]}]) == ("", b"\x89PNG fake")
	assert decisions.parse_request({"input": messages, "questions": [{"name": "hot_dog", "type": "predicate", "instructions": "Hot dog?"}]})[2]["image"] == b"\x89PNG fake"
	for part, message in [
		({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}, "aren't fetched"),
		({"type": "image_url", "image_url": {"url": "data:image/png,raw"}}, "base64"),
		({"type": "input_image", "image_url": "not base64!"}, "valid base64"),
	]:
		with pytest.raises(DecisionError, match=message):
			decisions.parse_input([{"role": "user", "content": [part]}])
	with pytest.raises(DecisionError, match="one image"):
		decisions.parse_input([{"role": "user", "content": "", "images": [png, png]}])
	with pytest.raises(DecisionError, match="text only"):
		decisions.state_text([{"role": "user", "content": "", "images": [png]}])
	q = {"name": "hot_dog", "type": "predicate", "instructions": "Hot dog?", "options": [True, False]}
	assert decisions.user_prompt("", q, image=True).startswith("Input: the image.\n\nQuestion: Hot dog?")
	assert decisions.user_prompt("menu", q, image=True).startswith("Input: the image, with this text:\nmenu\n")


def fake_model(name, questions=QUESTIONS, temperature=1.0, lean=1.0):
	"""A DecisionModel whose reader favors the last option by `lean`."""
	reader = FakeReader()
	reader.score = lambda state, qs, adapter=True, samples=1, image=None: (
		[[0.0] * (len(q["options"]) - 1) + [lean] for q in qs], {"input_tokens": 10, "cached_tokens": 4})
	return DecisionModel(reader, name, list(questions), temperature)


def test_serving_routes_ensembles_feedback_and_stats(tmp_path):
	from llmzempic.decision_monitor import DecisionLog, label_counts
	from llmzempic.decision_server import Ensemble, Route, parse_named, parse_route, unique_names

	assert parse_named(["a=/runs/1", "/runs/2", "C:\\x=y"]) == [("a", "/runs/1"), (None, "/runs/2"), (None, "C:\\x=y")]
	assert parse_route("live=a:90,b:10") == ("live", [("a", 90.0), ("b", 10.0)])
	assert parse_route("avg=a,b", ensemble=True) == ("avg", [("a", 1.0), ("b", 1.0)])
	for bad in ("live", "live=a", "live=a:0,b:1", "live=a:x,b"):
		with pytest.raises(DecisionError):
			parse_route(bad)
	a, b = fake_model("triage", lean=2.0), fake_model("triage", lean=-2.0)
	assert [m.name for m in unique_names([a, b])] == ["triage", "triage-2"]
	both = Ensemble("avg", [a, b])
	assert [q["name"] for q in both.questions] == ["intent", "urgency", "is_toxic"]
	route = Route("live", [a, b], [90, 10])
	assert route.pick("user-1") is route.pick("user-1")
	assert {route.pick(f"user-{i}").name for i in range(200)} == {"triage", "triage-2"}

	a.labels = label_counts([{"output": {"intent": "billing", "is_toxic": False}}] * 3, QUESTIONS)
	assert a.labels["intent"] == {"billing": 3, "technical_support": 0, "sales": 0, "other": 0} and a.labels["is_toxic"]["No"] == 3
	logs = {"triage": DecisionLog(tmp_path / "serve")}
	client = TestClient(create_decision_app([a, b, both], [route], logs))
	assert client.get("/v1/models").json()["default"] == "live"
	averaged = client.post("/v1/decisions", json={"model": "avg", "input": "x", "questions": [{"name": "is_toxic"}]}).json()
	assert averaged["answers"][0]["probability"] == pytest.approx(0.5) and averaged["usage"]["input_tokens"] == 20
	sticky = [client.post("/v1/decisions", json={"input": "hello", "user": "u7"}).json() for _ in range(3)]
	assert len({r["model"] for r in sticky}) == 1 and sticky[0]["route"] == "live"
	ids = []
	for i in range(4):
		r = client.post("/v1/decisions", json={"model": "triage", "input": f"bill {i}"}).json()
		assert r["answers"][0]["choice"] == "other"
		ids.append(r["id"])
	assert client.post("/v1/feedback", json={"id": ids[0], "answers": {"intent": "billing", "is_toxic": False}}).json() == {"ok": True}
	assert client.post("/v1/feedback", json={"id": ids[1], "answers": {"intent": "other"}}).status_code == 200
	assert client.post("/v1/feedback", json={"id": "dec_unknown", "answers": {"intent": "other"}}).status_code == 404
	assert client.post("/v1/feedback", json={"id": ids[2]}).status_code == 400
	assert client.post("/v1/decisions", json={"model": "nope", "input": "x"}).status_code == 400

	stats = client.get("/v1/stats", params={"model": "triage"}).json()
	logged = stats["count"]
	assert logged >= 4 and stats["feedback"] == 2
	intent = next(q for q in stats["questions"] if q["name"] == "intent")
	assert intent["answers"]["other"] == logged and intent["checked"] == 2 and intent["accuracy"] == 0.5
	assert intent["drift_tvd"] == 1.0 and intent["drifted"]
	assert stats["days"][-1]["count"] == logged

	from llmzempic.decision_monitor import feedback_examples
	examples = feedback_examples(tmp_path / "serve", QUESTIONS)
	assert [e["id"] for e in examples] == ids[:2]
	assert examples[0]["output"] == {"intent": "billing", "urgency": "critical", "is_toxic": False}
	assert examples[1]["input"] == "bill 1" and examples[1]["output"]["intent"] == "other"


def test_drift_measures_how_far_answers_moved():
	from llmzempic.decision_monitor import drift

	assert drift({"a": 50, "b": 50}, {"a": 50, "b": 50}) == (0.0, 0.0)
	tvd, psi = drift({"a": 90, "b": 10}, {"a": 50, "b": 50})
	assert tvd == 0.4 and psi > 0.2
	assert drift({}, {"a": 1}) == (None, None)


def test_run_monitor_reads_serve_logs_and_adds_feedback_examples(env):  # noqa: F811
	from llmzempic.decision_monitor import DecisionLog
	from llmzempic.fileio import write_json

	client, ws, jobs = env
	pid = client.post("/api/projects", json={"name": "Triage"}).json()["id"]
	client.put(f"/api/projects/{pid}", json={"config": {"answer_mode": "decisions", "schema": SCHEMA}})
	examples = [{"input": f"text {i}", "output": {"intent": "billing", "urgency": "low", "is_toxic": False}} for i in range(12)]
	client.put(f"/api/projects/{pid}/examples", json={"examples": examples})
	checked(client, pid)
	run_id = client.post(f"/api/projects/{pid}/runs").json()["id"]
	run_dir = ws.get(pid).runs_dir / run_id
	assert client.get(f"/api/projects/{pid}/runs/{run_id}/monitor").json()["stats"] is None
	write_json(run_dir / "calibration.json", {"temperature": 1.0, "labels": {"intent": {"billing": 12, "technical_support": 0, "sales": 0, "other": 0}}})
	log = DecisionLog(run_dir / "serve")
	response = decisions.build_response("triage", QUESTIONS, [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0]], 12)
	log.record(response, 40.0, text="where is my refund")
	log.feedback(response["id"], {"intent": "billing", "is_toxic": True})
	assert client.get(f"/api/projects/{pid}/runs").json()["runs"][0]["monitored"]
	monitor = client.get(f"/api/projects/{pid}/runs/{run_id}/monitor").json()
	assert monitor["examples"] == 1 and monitor["stats"]["count"] == 1
	assert next(q for q in monitor["stats"]["questions"] if q["name"] == "intent")["drift_tvd"] == 0.0
	assert client.post(f"/api/projects/{pid}/runs/{run_id}/monitor/examples").json() == {"added": 1}
	assert client.post(f"/api/projects/{pid}/runs/{run_id}/monitor/examples").json() == {"added": 0}
	added = client.get(f"/api/projects/{pid}/examples", params={"search": "refund"}).json()["examples"]
	assert added[0]["output"] == {"intent": "billing", "urgency": "medium", "is_toxic": True}


def test_mcp_server_exposes_a_decide_tool():
	import io

	from llmzempic.decision_mcp import handle, run_stdio

	served = fake_model("triage")
	init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}, served)
	assert init["result"]["protocolVersion"] == "2025-03-26" and "tools" in init["result"]["capabilities"]
	assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, served) is None
	tool = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, served)["result"]["tools"][0]
	assert tool["name"] == "decide" and "intent (choice: billing" in tool["description"] and tool["inputSchema"]["required"] == ["input"]
	call = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "decide", "arguments": {"input": "refund", "questions": [{"name": "intent"}]}}}, served)
	assert not call["result"]["isError"] and call["result"]["structuredContent"]["answers"][0]["choice"] == "other"
	assert json.loads(call["result"]["content"][0]["text"])[0]["name"] == "intent"
	bad = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "decide", "arguments": {}}}, served)
	assert bad["result"]["isError"] and "input" in bad["result"]["content"][0]["text"]
	assert handle({"jsonrpc": "2.0", "id": 5, "method": "nope"}, served)["error"]["code"] == -32601
	out = io.StringIO()
	run_stdio(served, io.StringIO('{"jsonrpc": "2.0", "id": 9, "method": "ping"}\nnot json\n\n'), out)
	assert [json.loads(line) for line in out.getvalue().splitlines()] == [
		{"jsonrpc": "2.0", "id": 9, "result": {}}, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}]


def test_export_folders_serve_on_their_own(tmp_path):
	from llmzempic.decision_engine import decision_source
	from llmzempic.exporter import copy_decision_model
	from llmzempic.fileio import write_json

	run = tmp_path / "run"
	(run / "adapter").mkdir(parents=True)
	(run / "adapter" / "adapter_config.json").write_text("{}")
	write_json(run / "calibration.json", {"temperature": 1.3, "labels": {"intent": {"billing": 2}}})
	spec = {
		"project_name": "Triage", "base_model": "org/model", "answer_mode": "decisions", "questions": QUESTIONS,
		"schema": SCHEMA, "system_prompt": decisions.system_prompt("Triage."), "instructions": "Triage.",
	}
	write_json(run / "run.json", spec)
	with pytest.raises(DecisionError, match="completed"):
		decision_source(run)
	write_json(run / "status.json", {"state": "completed"})
	assert decision_source(run)[0]["questions"] == QUESTIONS
	export = run / "export"
	export.mkdir()
	copy_decision_model(run, export)
	write_deploy_files(export, spec, "triage", None, 4096, None)
	loaded, adapter = decision_source(export)
	assert adapter == export / "adapter" and (adapter / "adapter_config.json").exists()
	assert read_json(export / "calibration.json")["temperature"] == 1.3
	assert loaded["base_model"] == "org/model" and loaded["instructions"] == "Triage."
	assert [decisions.user_prompt("x", q) for q in loaded["questions"]] == [decisions.user_prompt("x", q) for q in QUESTIONS]
	with pytest.raises(DecisionError, match="neither"):
		decision_source(tmp_path)
