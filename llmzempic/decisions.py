"""Decision models: typed questions answered with calibrated probabilities instead of generated text.

In a project whose answer mode is "decisions", every top-level output field is a question: an option
list is a choice question (a score question when its levels are ordered, SCORE_KEY), a yes/no field a
predicate. Each question is asked on its own: the prompt numbers the options and the model answers
with one number, so the probability of every option comes from a single batched forward pass over the
option numbers (no generation). Numbers scale to hundreds of options where letters run out.

Pure Python: the UI server, the API request checks and the metrics run without PyTorch.
"""

import math
import time
import uuid

from .schema_tools import compact_json

SCORE_KEY = "x-score"
TYPES = ("choice", "score", "predicate")
TYPE_ALIASES = {
	"multiple_choice": "choice", "classification": "choice", "rating": "score", "scale": "score",
	"bool": "predicate", "boolean": "predicate", "bernoulli": "predicate", "noul": "predicate", "binary": "predicate",
}
MAX_OPTIONS = 255
MAX_LEVELS = 10
MAX_QUESTIONS = 32
MAX_INPUT_CHARS = 200_000
YES_NO = ["Yes", "No"]
# Below this many scored answers a fitted temperature mostly fits noise; answers stay uncalibrated.
MIN_CALIBRATION = 50

DECISION_HEADER = "You answer one question about the input at a time. Reply with only the number of the best option."


class DecisionError(ValueError):
	pass


def is_decision_spec(spec_or_config):
	return (spec_or_config or {}).get("answer_mode") == "decisions"


def _types(prop):
	t = prop.get("type")
	return t if isinstance(t, list) else [t] if t else []


def schema_questions(schema):
	"""(questions, problems) for a decisions-mode schema: every top-level field must be a question."""
	props = schema.get("properties") if isinstance(schema, dict) else None
	if not isinstance(props, dict) or not props:
		return [], ["Decision models need at least one output field: an option list or a yes/no field."]
	questions, problems = [], []
	for name, prop in props.items():
		if not isinstance(prop, dict):
			problems.append(f'"{name}" isn\'t a schema object.')
			continue
		description = prop.get("description") if isinstance(prop.get("description"), str) else ""
		if isinstance(prop.get("enum"), list):
			options = prop["enum"]
			kind = "score" if prop.get(SCORE_KEY) else "choice"
			if not all(isinstance(v, str) for v in options):
				problems.append(f'"{name}": decision options must be text and can\'t be empty (null).')
			elif len(set(options)) != len(options):
				problems.append(f'"{name}" lists an option twice.')
			elif len(options) < 2:
				problems.append(f'"{name}" needs at least 2 options.')
			elif kind == "score" and len(options) > MAX_LEVELS:
				problems.append(f'"{name}": a score has 2 to {MAX_LEVELS} levels.')
			elif len(options) > MAX_OPTIONS:
				problems.append(f'"{name}" has more than {MAX_OPTIONS} options.')
			else:
				questions.append({"name": name, "type": kind, "instructions": description, "options": list(options)})
		elif _types(prop) == ["boolean"]:
			questions.append({"name": name, "type": "predicate", "instructions": description, "options": [True, False]})
		else:
			problems.append(
				f'"{name}" isn\'t an option list or a yes/no field. Decision models answer only those; '
				"remove it or switch the answer back to JSON."
			)
	return questions, problems


def humanize(name):
	words = str(name).replace("_", " ").replace("-", " ").strip()
	return words[:1].upper() + words[1:]


def question_text(question):
	text = (question.get("instructions") or "").strip()
	if text:
		return text
	name = str(question["name"]).replace("_", " ").replace("-", " ").strip()
	if question["type"] == "choice":
		return f"Which {name} fits the input?"
	if question["type"] == "score":
		return f"Rate the {name} of the input."
	return f"{humanize(name)}?"


def option_labels(question):
	return list(YES_NO) if question["type"] == "predicate" else [str(o) for o in question["options"]]


def option_ids(count):
	"""Answer numbers, zero-padded to one width so that no number is the start of another."""
	width = len(str(count))
	return [str(i + 1).zfill(width) for i in range(count)]


def system_prompt(instructions=""):
	return "\n\n".join(p for p in ((instructions or "").strip(), DECISION_HEADER) if p)


def user_prompt(state, question, order=None):
	"""The question about the input, its options numbered in display order (order[pos] = option index)."""
	labels = option_labels(question)
	order = order or list(range(len(labels)))
	ids = option_ids(len(labels))
	lines = "\n".join(f"{ids[pos]}. {labels[i]}" for pos, i in enumerate(order))
	header = "Levels, from lowest to highest:" if question["type"] == "score" else "Options:"
	return f"Input:\n{state}\n\nQuestion: {question_text(question)}\n{header}\n{lines}\n\nAnswer with the number of one option."


def shuffled_order(question, rng=None):
	"""Choice options are shown in a random order during training so the model reads the list instead of
	memorizing positions; scores keep their order and predicates their Yes/No."""
	order = list(range(len(question["options"])))
	if rng is not None and question["type"] == "choice":
		rng.shuffle(order)
	return order


def gold_index(question, value):
	if question["type"] == "predicate":
		return {True: 0, False: 1}.get(value) if isinstance(value, bool) else None
	if isinstance(value, str) and value in question["options"]:
		return question["options"].index(value)
	return None


def gold_items(output, questions):
	"""(question index, option index) for every question the expected output answers."""
	if not isinstance(output, dict):
		return []
	items = []
	for qi, q in enumerate(questions):
		if q["name"] in output:
			gold = gold_index(q, output[q["name"]])
			if gold is not None:
				items.append((qi, gold))
	return items


def answer_id(question, gold, order):
	return option_ids(len(question["options"]))[order.index(gold)]


def api_question(question):
	"""A question in the Decisions API request shape."""
	out = {"name": question["name"], "type": question["type"], "instructions": question_text(question)}
	if question["type"] == "choice":
		out["choices"] = list(question["options"])
	elif question["type"] == "score":
		out["levels"] = list(question["options"])
	return out


# API requests

def _messages_text(messages):
	lines = []
	for m in messages:
		content = m.get("content")
		if isinstance(content, list):
			parts = []
			for part in content:
				kind = part.get("type") if isinstance(part, dict) else None
				if kind in ("text", "input_text"):
					parts.append(str(part.get("text") or ""))
				elif kind in ("image", "image_url", "input_image"):
					raise DecisionError("This decision model reads text only; images in the input aren't supported yet.")
				else:
					raise DecisionError("Message content parts must be text.")
			content = "\n".join(parts)
		elif not isinstance(content, str):
			content = compact_json(content)
		lines.append(f"{m['role']}: {content}")
	return "\n".join(lines)


def state_text(value):
	"""The input as text: a string as is, chat messages as "role: content" lines, other JSON compactly."""
	if value is None:
		raise DecisionError('Add an "input" to decide on.')
	if isinstance(value, str):
		text = value
	elif isinstance(value, list) and value and all(isinstance(m, dict) and "role" in m for m in value):
		text = _messages_text(value)
	else:
		text = compact_json(value)
	if not text.strip():
		raise DecisionError("The input is empty.")
	if len(text) > MAX_INPUT_CHARS:
		raise DecisionError(f"The input is longer than {MAX_INPUT_CHARS:,} characters.")
	return text


def _option_list(item, key, kind, name):
	values = item.get(key)
	if not isinstance(values, list):
		raise DecisionError(f'Question "{name}" needs "{key}": a list of options.')
	if not all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in values):
		raise DecisionError(f'Question "{name}": options must be text.')
	values = [str(v) for v in values]
	if any(not v.strip() for v in values) or len(set(values)) != len(values):
		raise DecisionError(f'Question "{name}": options must be non-empty and different.')
	limit = MAX_LEVELS if kind == "score" else MAX_OPTIONS
	if not 2 <= len(values) <= limit:
		raise DecisionError(f'Question "{name}" needs 2 to {limit} {"levels" if kind == "score" else "choices"}.')
	return values


def parse_question(item, index, known):
	if not isinstance(item, dict):
		raise DecisionError("Each question must be a JSON object.")
	name = str(item.get("name") or f"question_{index + 1}")
	base = known.get(name)
	kind = str(item.get("type") or (base or {}).get("type") or "").lower()
	kind = TYPE_ALIASES.get(kind, kind)
	if kind not in TYPES:
		raise DecisionError(f'Question "{name}" needs a "type": choice, score or predicate.')
	instructions = item.get("instructions")
	if instructions is None and base:
		instructions = base["instructions"] or question_text(base)
	if not isinstance(instructions, str) or not instructions.strip():
		raise DecisionError(f'Question "{name}" needs "instructions".')
	same_kind = base and base["type"] == kind
	if kind == "predicate":
		options = [True, False]
	elif kind == "choice":
		options = _option_list(item, "choices", kind, name) if "choices" in item or not same_kind else list(base["options"])
	else:
		options = _option_list(item, "levels", kind, name) if "levels" in item or not same_kind else list(base["options"])
	# The model was trained on this question (its options at least): its calibration applies.
	trained = bool(same_kind and base["options"] == options)
	return {"name": name, "type": kind, "instructions": instructions.strip(), "options": options, "trained": trained}


def parse_request(body, trained=None):
	"""(input text, questions, min confidence) from a Decisions API request. Questions may be left out
	(all trained questions) or name a trained question and leave out the rest of its definition."""
	if not isinstance(body, dict):
		raise DecisionError("The request body must be a JSON object.")
	state = state_text(body.get("input"))
	known = {q["name"]: q for q in trained or []}
	raw = body.get("questions")
	if raw is None:
		if not known:
			raise DecisionError('Add "questions" to answer.')
		raw = [{"name": name} for name in known]
	if not isinstance(raw, list) or not raw:
		raise DecisionError('"questions" must be a non-empty list.')
	if len(raw) > MAX_QUESTIONS:
		raise DecisionError(f"Ask at most {MAX_QUESTIONS} questions per request.")
	questions = [parse_question(item, i, known) for i, item in enumerate(raw)]
	if len({q["name"] for q in questions}) != len(questions):
		raise DecisionError("Question names must be different.")
	threshold = body.get("min_confidence")
	if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
		raise DecisionError('"min_confidence" must be a number from 0 to 1.')
	return state, questions, threshold


# Answers

def softmax(scores, temperature=1.0):
	scaled = [s / temperature for s in scores]
	top = max(scaled)
	exps = [math.exp(s - top) for s in scaled]
	total = sum(exps)
	return [e / total for e in exps]


def argmax(values):
	return max(range(len(values)), key=values.__getitem__)


def predicted_value(question, probs):
	"""The answer as the project's output value: an option, a level, or true/false."""
	if question["type"] == "predicate":
		return probs[0] >= 0.5
	return question["options"][argmax(probs)]


def build_answer(question, probs, threshold=None):
	"""One answer in the Decisions API shape. Confidence is the probability of the returned answer; below
	threshold the answer is withheld (refused) but its probabilities are still reported."""
	top = argmax(probs)
	confidence = probs[top]
	answer = {"name": question["name"], "type": question["type"]}
	refused = threshold is not None and confidence < threshold
	if question["type"] == "predicate":
		answer["probability"] = round(probs[0], 4)
	else:
		labels = option_labels(question)
		if question["type"] == "choice":
			answer["choice"] = None if refused else labels[top]
		else:
			expected = sum(p * (i + 1) for i, p in enumerate(probs))
			answer["score"] = None if refused else round(expected, 4)
			answer["level"] = None if refused else labels[top]
		answer["probabilities"] = {label: round(p, 4) for label, p in zip(labels, probs)}
	answer["confidence"] = round(confidence, 4)
	if refused:
		answer["refused"] = True
	return answer


def question_temperature(question, temperature):
	return temperature if question.get("trained", True) else 1.0


def build_response(model, questions, scores, input_tokens, temperature=1.0, threshold=None):
	answers = [
		build_answer(q, softmax(s, question_temperature(q, temperature)), threshold)
		for q, s in zip(questions, scores)
	]
	return {
		"id": f"dec_{uuid.uuid4().hex[:24]}",
		"object": "decision",
		"created": int(time.time()),
		"model": model,
		"answers": answers,
		"usage": {"input_tokens": int(input_tokens), "output_tokens": 0},
	}


# Metrics

def ece(confidences, correct, bins=10):
	"""Expected calibration error of the top answer: the gap between confidence and accuracy, per bin."""
	n = len(confidences)
	if not n:
		return 0.0
	totals = [[0, 0.0, 0] for _ in range(bins)]
	for conf, ok in zip(confidences, correct):
		b = totals[min(bins - 1, int(conf * bins))]
		b[0] += 1
		b[1] += conf
		b[2] += 1 if ok else 0
	return sum(abs(c - k) for count, c, k in totals if count) / n


def _sample_stats(probs, gold):
	top = argmax(probs)
	return {
		"correct": top == gold,
		"confidence": probs[top],
		"nll": -math.log(max(probs[gold], 1e-12)),
		"brier": sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(probs)),
	}


def summarize(stats):
	n = len(stats)
	if not n:
		return {"count": 0, "accuracy": 0.0, "nll": 0.0, "brier": 0.0, "ece": 0.0}
	return {
		"count": n,
		"accuracy": round(sum(1 for s in stats if s["correct"]) / n, 4),
		"nll": round(sum(s["nll"] for s in stats) / n, 4),
		"brier": round(sum(s["brier"] for s in stats) / n, 4),
		"ece": round(ece([s["confidence"] for s in stats], [s["correct"] for s in stats]), 4),
	}


def mean_nll(samples, temperature):
	return sum(-math.log(max(softmax(s["scores"], temperature)[s["gold"]], 1e-12)) for s in samples) / max(1, len(samples))


def fit_temperature(samples):
	"""Post-hoc temperature scaling: the one temperature that minimizes held-out NLL (log-space grid, then refined)."""
	if not samples:
		return 1.0
	grid = [math.exp(math.log(0.05) + i * (math.log(20) - math.log(0.05)) / 80) for i in range(81)]
	best = min(grid, key=lambda t: mean_nll(samples, t))
	fine = [best * math.exp((i - 10) * 0.0075) for i in range(21)]
	return round(min(fine, key=lambda t: mean_nll(samples, t)), 4)


def calibration_report(samples, questions):
	"""samples: [{"question": index, "scores": [log-score per option], "gold": option index}].

	Raw metrics use the model's probabilities as read. Calibrated metrics are cross-fitted (temperature fit
	on one half of the samples, scored on the other) so they aren't measured on the data they were fit on;
	the temperature saved for serving is fit on all of them.
	"""
	raw = [_sample_stats(softmax(s["scores"]), s["gold"]) for s in samples]
	fitted = len(samples) >= MIN_CALIBRATION
	temperature = fit_temperature(samples) if fitted else 1.0
	if fitted:
		halves = [samples[0::2], samples[1::2]]
		temps = [fit_temperature(halves[1]), fit_temperature(halves[0])]
		calibrated = [None] * len(samples)
		for half, t in enumerate(temps):
			for j, s in enumerate(halves[half]):
				calibrated[half + 2 * j] = _sample_stats(softmax(s["scores"], t), s["gold"])
	else:
		calibrated = raw
	per_question = []
	for qi, q in enumerate(questions):
		idx = [i for i, s in enumerate(samples) if s["question"] == qi]
		if not idx:
			continue
		# Temperature never changes the top answer, so accuracy is the same before and after.
		row = {"name": q["name"], "type": q["type"], **summarize([calibrated[i] for i in idx])}
		if q["type"] == "score":
			levels = range(1, len(q["options"]) + 1)
			errors = [abs(sum(p * k for p, k in zip(softmax(samples[i]["scores"], temperature), levels)) - (samples[i]["gold"] + 1)) for i in idx]
			row["mae"] = round(sum(errors) / len(errors), 4)
		per_question.append(row)
	return {
		"temperature": temperature,
		"calibrated": fitted,
		**summarize(raw),
		"after_calibration": summarize(calibrated),
		"questions": per_question,
	}


def percentile(values, share):
	if not values:
		return None
	ordered = sorted(values)
	return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]
