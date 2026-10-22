"""Scoring generated outputs against expected outputs and the schema."""

import json

from .dataset import label_fields, value_key
from .schema_tools import check_schema, validate_output


def parse_generated(text):
	"""Decode a model response into JSON. Returns (value, ok)."""
	text = (text or "").strip()
	if text.startswith("```"):
		text = text.strip("`")
		if text.lower().startswith("json"):
			text = text[4:]
		text = text.strip()
	try:
		return json.loads(text), True
	except ValueError:
		pass
	decoder = json.JSONDecoder()
	for i, ch in enumerate(text):
		if ch in "{[":
			try:
				return decoder.raw_decode(text[i:])[0], True
			except ValueError:
				continue
	return None, False


def score_predictions(rows, generations, schema):
	"""rows: [{"input", "output", "image"?}]; generations: model texts in the same order."""
	items = []
	field_hits = {}
	for row, text in zip(rows, generations):
		value, ok = parse_generated(text)
		errors = validate_output(value, schema) if ok else ["Response is not valid JSON"]
		exact = ok and value == row["output"]
		expected = row["output"]
		if isinstance(expected, dict):
			for key, want in expected.items():
				hit = ok and isinstance(value, dict) and value.get(key) == want
				total, good = field_hits.get(key, (0, 0))
				field_hits[key] = (total + 1, good + (1 if hit else 0))
		items.append({
			"input": row["input"],
			"image": row.get("image"),
			"expected": expected,
			"generated": text,
			"json_valid": ok,
			"schema_valid": ok and not errors,
			"exact_match": exact,
			"errors": errors,
		})

	n = len(items)

	def rate(key):
		return round(sum(1 for i in items if i[key]) / n, 4) if n else 0.0

	return {
		"count": n,
		"json_valid": rate("json_valid"),
		"schema_valid": rate("schema_valid"),
		"exact_match": rate("exact_match"),
		"field_accuracy": {k: round(g / t, 4) for k, (t, g) in field_hits.items()},
		"items": items,
	}


def _label_rows(items, name, kind, values):
	allowed = {value_key(v): v for v in values}
	rows = {k: {"value": v, "expected": 0, "predicted": 0, "correct": 0} for k, v in allowed.items()}
	confusions = {}
	for item in items:
		expected = item.get("expected")
		if not isinstance(expected, dict) or name not in expected:
			continue
		value, ok = parse_generated(item.get("generated"))
		has = ok and isinstance(value, dict) and name in value
		if kind == "tags":
			want = {value_key(v) for v in expected[name]} & allowed.keys() if isinstance(expected[name], list) else set()
			got = {value_key(v) for v in value[name]} & allowed.keys() if has and isinstance(value[name], list) else set()
			for key in want:
				rows[key]["expected"] += 1
			for key in got:
				rows[key]["predicted"] += 1
			for key in want & got:
				rows[key]["correct"] += 1
			continue
		want = value_key(expected[name])
		if want not in allowed:
			continue
		rows[want]["expected"] += 1
		got = value_key(value[name]) if has else None
		if got in allowed:
			rows[got]["predicted"] += 1
		if got == want:
			rows[want]["correct"] += 1
		else:
			confusions[(want, got)] = confusions.get((want, got), 0) + 1
	mixups = [
		{"expected": allowed[want], "predicted": allowed.get(got), "unknown": got not in allowed, "count": n}
		for (want, got), n in sorted(confusions.items(), key=lambda kv: -kv[1])[:10]
	]
	return {
		"field": name,
		"kind": kind,
		"values": [r for r in rows.values() if r["expected"] or r["predicted"]],
		"confusions": mixups,
	}


def label_breakdown(items, schema):
	"""Per-label scores for enum, boolean and tag fields: how often each expected value was
	recognized, how often each predicted value was right, and the most common mix-ups."""
	if check_schema(schema):
		return []
	fields = [_label_rows(items, *field) for field in label_fields(schema)]
	return [f for f in fields if f["values"]]

