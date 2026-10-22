"""Scoring generated outputs against expected outputs and the schema."""

import json

from .schema_tools import validate_output


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
	"""rows: [{"input", "output"}]; generations: model texts in the same order."""
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
