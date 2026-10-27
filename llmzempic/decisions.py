"""Decision models: typed questions answered with calibrated probabilities instead of generated text.

In a project whose answer mode is "decisions", every top-level output field is a question: an option
list is a choice question (a score question when its levels are ordered, SCORE_KEY), a yes/no field a
predicate. Each question is asked on its own: the prompt numbers the options and the model answers
with one number, so the probability of every option comes from a single batched forward pass over the
option numbers (no generation). Numbers scale to hundreds of options where letters run out.

Pure Python: the UI server, the API request checks and the metrics run without PyTorch.
"""

import base64
import binascii
import math
import random
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
MAX_SAMPLES = 8
MAX_INPUT_CHARS = 200_000
YES_NO = ["Yes", "No"]
# Below this many scored answers a fitted temperature mostly fits noise; answers stay uncalibrated.
MIN_CALIBRATION = 50
# Share of training sequences that show a choice question with only some of its options (the gold one
# always among them), so the model also answers well when a request lists fewer choices.
SUBSET_SHARE = 0.25
# Accuracy that min_confidence "auto" aims for, per question, from the held-out precision curve.
DEFAULT_TARGET = 0.9

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


def user_prompt(state, question, order=None, image=False):
	"""The question about the input, its options numbered in display order (order[pos] = option index; an
	order may list only some of the options). With image, the input is the image in the same message, with
	state as its optional text."""
	labels = option_labels(question)
	order = order or list(range(len(labels)))
	ids = option_ids(len(order))
	lines = "\n".join(f"{ids[pos]}. {labels[i]}" for pos, i in enumerate(order))
	header = "Levels, from lowest to highest:" if question["type"] == "score" else "Options:"
	if image:
		text = (state or "").strip()
		subject = f"Input: the image, with this text:\n{text}" if text else "Input: the image."
	else:
		subject = f"Input:\n{state}"
	return f"{subject}\n\nQuestion: {question_text(question)}\n{header}\n{lines}\n\nAnswer with the number of one option."


def shuffled_order(question, rng=None, gold=None, subset_share=0.0):
	"""Choice options are shown in a random order during training so the model reads the list instead of
	memorizing positions, and sometimes only some of them (always with the gold one); scores keep their
	order and predicates their Yes/No."""
	order = list(range(len(question["options"])))
	if rng is None or question["type"] != "choice":
		return order
	rng.shuffle(order)
	if gold is not None and len(order) > 2 and rng.random() < subset_share:
		kept = {gold, *[i for i in order if i != gold][:rng.randint(1, len(order) - 2)]}
		order = [i for i in order if i in kept]
	return order


def sample_orders(question, samples):
	"""Option orders to read a question in: as listed, then (choice questions, samples > 1) shuffled the
	same way for every request, so a repeated request gets the same answer."""
	identity = list(range(len(question["options"])))
	if samples <= 1 or question["type"] != "choice":
		return [identity]
	rng = random.Random(f"{question['name']}:{len(identity)}")
	orders, seen = [identity], {tuple(identity)}
	for _ in range(samples * 4):
		if len(orders) >= samples:
			break
		order = identity[:]
		rng.shuffle(order)
		if tuple(order) not in seen:
			seen.add(tuple(order))
			orders.append(order)
	return orders


def combine_samples(score_sets):
	"""One log-probability per option from reads in several orders: the log of their mean probability."""
	if len(score_sets) == 1:
		return list(score_sets[0])
	probs = [softmax(s) for s in score_sets]
	return [math.log(max(sum(p[i] for p in probs) / len(probs), 1e-30)) for i in range(len(probs[0]))]


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
	return option_ids(len(order))[order.index(gold)]


def api_question(question):
	"""A question in the Decisions API request shape."""
	out = {"name": question["name"], "type": question["type"], "instructions": question_text(question)}
	if question["type"] == "choice":
		out["choices"] = list(question["options"])
	elif question["type"] == "score":
		out["levels"] = list(question["options"])
	return out


def questions_schema(questions):
	"""The output schema a list of questions answers: an option list per choice or score, yes/no per predicate."""
	properties = {}
	for q in questions:
		prop = {"type": "boolean"} if q["type"] == "predicate" else {"type": "string", "enum": list(q["options"])}
		if q["type"] == "score":
			prop[SCORE_KEY] = True
		if q.get("instructions"):
			prop["description"] = q["instructions"]
		properties[q["name"]] = prop
	return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def summarize_at(samples, temperature):
	"""Metrics of samples read at a fixed temperature (the one a run serves with)."""
	return {"temperature": temperature, **summarize([_sample_stats(softmax(s["scores"], temperature), s["gold"]) for s in samples])}


def from_api_question(item):
	"""A question as decisions.json / the API lists it, back in the internal shape."""
	kind = item["type"]
	options = [True, False] if kind == "predicate" else list(item["choices"] if kind == "choice" else item["levels"])
	return {"name": item["name"], "type": kind, "instructions": item.get("instructions") or "", "options": options}


# API requests

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def decode_image(part):
	"""Image bytes from a content part: OpenAI image_url / input_image or an Ollama-style base64 string.
	Only data URLs and plain base64 are read; the server never fetches URLs or opens local files."""
	value = part.get("image_url") if part.get("image_url") is not None else part.get("image")
	if isinstance(value, dict):
		value = value.get("url")
	if not isinstance(value, str) or not value:
		raise DecisionError("An image part needs a base64 data URL.")
	if value.startswith(("http://", "https://")):
		raise DecisionError("Send images as base64 data URLs; image links aren't fetched.")
	if value.startswith("data:"):
		header, _, value = value.partition(",")
		if ";base64" not in header:
			raise DecisionError("Image data URLs must be base64 encoded.")
	try:
		data = base64.b64decode(value, validate=True)
	except (binascii.Error, ValueError):
		raise DecisionError("The image isn't valid base64.")
	if not data or len(data) > MAX_IMAGE_BYTES:
		raise DecisionError(f"Images must be 1 byte to {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
	return data


def _messages_input(messages):
	lines, images = [], []
	for m in messages:
		content = m.get("content")
		if isinstance(content, list):
			parts = []
			for part in content:
				kind = part.get("type") if isinstance(part, dict) else None
				if kind in ("text", "input_text"):
					parts.append(str(part.get("text") or ""))
				elif kind in ("image", "image_url", "input_image"):
					images.append(decode_image(part))
				else:
					raise DecisionError("Message content parts must be text or images.")
			content = "\n".join(p for p in parts if p)
		elif not isinstance(content, str):
			content = compact_json(content)
		for image in m.get("images") or ():  # Ollama-style messages
			images.append(decode_image({"image": image}))
		if content:
			lines.append(f"{m['role']}: {content}")
	return "\n".join(lines), images


def parse_input(value):
	"""(text, image bytes or None): a string as is, chat messages as "role: content" lines plus their image,
	other JSON compactly. Decision models read one image per input."""
	if value is None:
		raise DecisionError('Add an "input" to decide on.')
	images = []
	if isinstance(value, str):
		text = value
	elif isinstance(value, list) and value and all(isinstance(m, dict) and "role" in m for m in value):
		text, images = _messages_input(value)
	else:
		text = compact_json(value)
	if len(images) > 1:
		raise DecisionError("Send one image per input.")
	if not text.strip() and not images:
		raise DecisionError("The input is empty.")
	if len(text) > MAX_INPUT_CHARS:
		raise DecisionError(f"The input is longer than {MAX_INPUT_CHARS:,} characters.")
	return text, images[0] if images else None


def state_text(value):
	"""The input as text, for models that read text only."""
	text, image = parse_input(value)
	if image is not None:
		raise DecisionError("This decision model reads text only; images in the input aren't supported.")
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
	# The model was trained on this question and these options (or some of them, as it also trains on option
	# subsets): its calibration applies.
	trained = bool(same_kind and (base["options"] == options or (kind == "choice" and set(options) <= set(base["options"]))))
	return {"name": name, "type": kind, "instructions": instructions.strip(), "options": options, "trained": trained}


def _fraction(body, key):
	value = body.get(key)
	if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1):
		raise DecisionError(f'"{key}" must be a number from 0 to 1.')
	return value


def parse_settings(body):
	"""Answer settings of a request: refusal threshold (a number, or "auto" for the per-question threshold
	that reached target_accuracy on held-out examples) and samples (option orders read per choice question)."""
	threshold = body.get("min_confidence")
	if threshold != "auto":
		threshold = _fraction(body, "min_confidence")
	target = _fraction(body, "target_accuracy")
	if target is not None and threshold is None:
		threshold = "auto"
	samples = body.get("samples", 1)
	if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= MAX_SAMPLES:
		raise DecisionError(f'"samples" must be a whole number from 1 to {MAX_SAMPLES}.')
	return {"min_confidence": threshold, "target_accuracy": target if target is not None else DEFAULT_TARGET, "samples": samples}


def parse_request(body, trained=None):
	"""(input text, questions, settings) from a Decisions API request. Questions may be left out (all
	trained questions) or name a trained question and leave out the rest of its definition. An image in
	the input is in settings["image"] (bytes)."""
	if not isinstance(body, dict):
		raise DecisionError("The request body must be a JSON object.")
	state, image = parse_input(body.get("input"))
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
	return state, questions, {**parse_settings(body), "image": image}


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


def question_threshold(question, settings, curves=None):
	"""The confidence below which this question's answer is refused, or None."""
	threshold = (settings or {}).get("min_confidence")
	if threshold != "auto":
		return threshold
	curve = (curves or {}).get(question["name"]) if question.get("trained", True) else None
	return auto_threshold(curve, settings.get("target_accuracy") or DEFAULT_TARGET) if curve else None


def usage_of(usage):
	usage = usage if isinstance(usage, dict) else {"input_tokens": usage}
	out = {"input_tokens": int(usage.get("input_tokens") or 0), "output_tokens": 0}
	if usage.get("cached_tokens"):
		out["input_tokens_details"] = {"cached_tokens": int(usage["cached_tokens"])}
	return out


def build_response(model, questions, scores, usage, temperature=1.0, settings=None, curves=None):
	"""scores: one log-probability per option per question. settings: parse_settings() of the request (or
	a plain min_confidence number); curves: per-question precision curves for "auto" thresholds."""
	if not isinstance(settings, dict):
		settings = {"min_confidence": settings}
	answers = [
		build_answer(q, softmax(s, question_temperature(q, temperature)), question_threshold(q, settings, curves))
		for q, s in zip(questions, scores)
	]
	return {
		"id": f"dec_{uuid.uuid4().hex[:24]}",
		"object": "decision",
		"created": int(time.time()),
		"model": model,
		"answers": answers,
		"usage": usage_of(usage),
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


def precision_curve(confidences, correct, points=40):
	"""[[threshold, precision, coverage]] for answers at least that confident, highest threshold first."""
	pairs = sorted(zip(confidences, correct), reverse=True)
	n = len(pairs)
	if not n:
		return []
	step = max(1, n // points)
	curve, hits = [], 0
	for i, (conf, ok) in enumerate(pairs, start=1):
		hits += 1 if ok else 0
		last = i == n or pairs[i][0] < conf
		if last and (i % step == 0 or i == n or i <= 3):
			# Rounded down, so every answer counted at a threshold is at least that confident.
			curve.append([math.floor(conf * 10_000) / 10_000, round(hits / i, 4), round(i / n, 4)])
	return curve


def auto_threshold(curve, target):
	"""Lowest threshold whose answers were at least target accurate on held-out examples (most answers kept),
	1.0 (refuse everything) when no threshold reached it."""
	best = None
	for threshold, precision, _ in curve:
		if precision >= target:
			best = threshold
	return 1.0 if best is None else best


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
		served = [_sample_stats(softmax(samples[i]["scores"], temperature), samples[i]["gold"]) for i in idx]
		row["curve"] = precision_curve([s["confidence"] for s in served], [s["correct"] for s in served])
		threshold = auto_threshold(row["curve"], DEFAULT_TARGET)
		kept = [s for s in served if s["confidence"] >= threshold]
		row["auto"] = {
			"target": DEFAULT_TARGET, "min_confidence": threshold, "coverage": round(len(kept) / len(served), 4),
			"accuracy": round(sum(1 for s in kept if s["correct"]) / len(kept), 4) if kept else None,
		}
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
