"""Watching decision models in production. `llmzempic serve` logs every decision (and any feedback on it) as
JSONL in the served folder's serve/ directory; the statistics here (shown by GET /v1/stats and the run view)
are the answer distribution and its drift from the training labels, confidence and refusal rates over time,
and, where feedback says what the right answer was, accuracy and calibration over time.

Pure Python: the UI server reads these logs without loading PyTorch.
"""

import hashlib
import json
import math
import threading
import time
from pathlib import Path

from . import decisions
from .fileio import append_jsonl

LOG = "decisions.jsonl"
FEEDBACK = "feedback.jsonl"
SERVE_DIR = "serve"
# PSI above this is the usual sign that what the model sees has shifted from what it was trained on.
DRIFT_PSI = 0.2


def top_value(answer):
	"""The model's most likely answer as an option label (Yes/No for predicates), refused or not."""
	if answer["type"] == "predicate":
		return "Yes" if answer["probability"] >= 0.5 else "No"
	probs = answer.get("probabilities") or {}
	return max(probs, key=probs.get) if probs else None


def answer_value(answer):
	"""The answer a decision gave, as an option label; None when refused."""
	return None if answer.get("refused") else top_value(answer)


def label_counts(rows, questions):
	"""{question: {option label: count}} of the expected answers, the reference for drift."""
	counts = {q["name"]: {label: 0 for label in decisions.option_labels(q)} for q in questions}
	for row in rows:
		for qi, gold in decisions.gold_items(row.get("output"), questions):
			q = questions[qi]
			counts[q["name"]][decisions.option_labels(q)[gold]] += 1
	return counts


class DecisionLog:
	"""Appends decisions and feedback for one served model. Inputs are logged (so corrected answers can
	become training examples) unless log_inputs is off; images are logged as a hash only."""

	def __init__(self, folder, log_inputs=True):
		self.folder = Path(folder)
		self.folder.mkdir(parents=True, exist_ok=True)
		self.log_inputs = log_inputs
		self.lock = threading.Lock()

	def record(self, response, latency_ms, route=None, text=None, image=None):
		row = {
			"id": response["id"],
			"time": response["created"],
			"model": response["model"],
			"route": route,
			"latency_ms": round(latency_ms, 1),
			"input_tokens": response["usage"]["input_tokens"],
			"answers": [
				{k: a[k] for k in ("name", "type", "choice", "level", "score", "probability", "probabilities", "confidence", "refused") if k in a}
				for a in response["answers"]
			],
		}
		if self.log_inputs and text is not None:
			row["input"] = text
		if image is not None:
			row["image_sha256"] = hashlib.sha256(image).hexdigest()
		with self.lock:
			append_jsonl(self.folder / LOG, row)

	def feedback(self, decision_id, answers):
		with self.lock:
			append_jsonl(self.folder / FEEDBACK, {"id": decision_id, "time": time.time(), "answers": answers})


def read_rows(path):
	path = Path(path)
	if not path.exists():
		return
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			try:
				yield json.loads(line)
			except ValueError:
				continue  # a line being written


def feedback_value(answer_type, value):
	"""Feedback's right answer as an option label, or None if it isn't one."""
	if answer_type == "predicate":
		return {True: "Yes", False: "No", "yes": "Yes", "no": "No", "Yes": "Yes", "No": "No"}.get(value) if isinstance(value, (bool, str)) else None
	return value if isinstance(value, str) else None


def read_feedback(folder):
	"""{decision id: {question: right answer}}, the latest feedback for each decision winning."""
	latest = {}
	for row in read_rows(Path(folder) / FEEDBACK):
		if isinstance(row.get("answers"), dict):
			latest.setdefault(row["id"], {}).update(row["answers"])
	return latest


def drift(observed, reference):
	"""(total variation distance, population stability index) between two {label: count} distributions."""
	labels = sorted(set(observed) | set(reference))
	total_o, total_r = sum(observed.values()), sum(reference.values())
	if not total_o or not total_r:
		return None, None
	tvd = psi = 0.0
	for label in labels:
		o = observed.get(label, 0) / total_o
		r = reference.get(label, 0) / total_r
		tvd += abs(o - r) / 2
		# A small floor keeps options never seen on one side from making PSI infinite.
		o, r = max(o, 1e-4), max(r, 1e-4)
		psi += (o - r) * math.log(o / r)
	return round(tvd, 4), round(psi, 4)


def _day(timestamp):
	return time.strftime("%Y-%m-%d", time.localtime(timestamp))


def monitor_stats(folder, reference=None, days=30):
	"""Statistics for one served model's log: totals, per question (answer distribution, drift from the
	training labels, mean confidence, refusal rate, accuracy and calibration from feedback), and per day."""
	folder = Path(folder)
	feedback = read_feedback(folder)
	questions, daily = {}, {}
	total = refused_any = 0
	latencies = []
	first = last = None
	for row in read_rows(folder / LOG):
		total += 1
		first = row["time"] if first is None else min(first, row["time"])
		last = row["time"] if last is None else max(last, row["time"])
		latencies.append(row.get("latency_ms") or 0)
		day = daily.setdefault(_day(row["time"]), {"count": 0, "confidence": 0.0, "answers": 0, "refused": 0, "checked": 0, "correct": 0})
		day["count"] += 1
		right = feedback.get(row["id"]) or {}
		any_refused = False
		for a in row["answers"]:
			q = questions.setdefault(a["name"], {
				"name": a["name"], "type": a["type"], "count": 0, "refused": 0, "confidence": 0.0, "answers": {},
				"checked": 0, "correct": 0, "feedback": [],
			})
			q["count"] += 1
			q["confidence"] += a["confidence"]
			day["answers"] += 1
			day["confidence"] += a["confidence"]
			value = answer_value(a)
			if value is None:
				q["refused"] += 1
				day["refused"] += 1
				any_refused = True
			else:
				q["answers"][value] = q["answers"].get(value, 0) + 1
			if a["name"] in right:
				want = feedback_value(a["type"], right[a["name"]])
				if want is not None:
					ok = top_value(a) == want
					q["checked"] += 1
					q["correct"] += 1 if ok else 0
					q["feedback"].append((a["confidence"], ok))
					day["checked"] += 1
					day["correct"] += 1 if ok else 0
		refused_any += 1 if any_refused else 0
	per_question = []
	for q in questions.values():
		row = {
			"name": q["name"], "type": q["type"], "count": q["count"],
			"refusal_rate": round(q["refused"] / q["count"], 4) if q["count"] else 0.0,
			"mean_confidence": round(q["confidence"] / q["count"], 4) if q["count"] else 0.0,
			"answers": dict(sorted(q["answers"].items(), key=lambda kv: -kv[1])),
			"checked": q["checked"],
		}
		ref = (reference or {}).get(q["name"])
		if ref:
			row["drift_tvd"], row["drift_psi"] = drift(q["answers"], ref)
			row["drifted"] = bool(row["drift_psi"] is not None and row["drift_psi"] > DRIFT_PSI)
			row["reference"] = ref
		if q["checked"]:
			row["accuracy"] = round(q["correct"] / q["checked"], 4)
			confidences = [c for c, _ in q["feedback"]]
			row["ece"] = round(decisions.ece(confidences, [ok for _, ok in q["feedback"]]), 4)
		per_question.append(row)
	series = []
	for day in sorted(daily)[-days:]:
		d = daily[day]
		series.append({
			"day": day, "count": d["count"],
			"mean_confidence": round(d["confidence"] / d["answers"], 4) if d["answers"] else None,
			"refusal_rate": round(d["refused"] / d["answers"], 4) if d["answers"] else None,
			"accuracy": round(d["correct"] / d["checked"], 4) if d["checked"] else None,
			"checked": d["checked"],
		})
	return {
		"count": total,
		"first": first,
		"last": last,
		"refusal_rate": round(refused_any / total, 4) if total else 0.0,
		"latency_ms": {"p50": decisions.percentile(latencies, 0.5), "p95": decisions.percentile(latencies, 0.95)} if latencies else None,
		"feedback": len(feedback),
		"questions": per_question,
		"days": series,
	}


def feedback_examples(folder, questions):
	"""Training examples from decisions that got feedback and whose input was logged: the feedback's
	answers, the model's own answers for questions feedback didn't cover. Keyed by decision id."""
	folder = Path(folder)
	feedback = read_feedback(folder)
	by_name = {q["name"]: q for q in questions}
	examples = []
	for row in read_rows(folder / LOG):
		right = feedback.get(row["id"])
		if right is None or not isinstance(row.get("input"), str) or row.get("image_sha256"):
			continue
		output = {}
		for a in row["answers"]:
			q = by_name.get(a["name"])
			if q is None:
				continue
			# Questions without feedback keep the model's top answer, refused or not (the example gets reviewed).
			label = feedback_value(a["type"], right[a["name"]]) if a["name"] in right else top_value(a)
			if label is None or label not in decisions.option_labels(q):
				continue
			output[q["name"]] = label == "Yes" if q["type"] == "predicate" else label
		if output:
			examples.append({"id": row["id"], "input": row["input"], "output": output})
	return examples
