"""JSON Schema helpers: validation, inference from examples, templates and CSV coercion."""

import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def compact_json(value):
	return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def check_schema(schema):
	"""Return a list of problems with the schema itself (empty when usable)."""
	if not isinstance(schema, dict):
		return ["The output structure must be a JSON Schema object."]
	try:
		Draft202012Validator.check_schema(schema)
	except SchemaError as e:
		return [f"Invalid JSON Schema: {e.message}"]
	if not schema:
		return ["The schema is empty - describe the expected output structure."]
	return []


def validate_output(value, schema):
	"""Validate one expected/generated output. Returns a list of readable errors."""
	if check_schema(schema):
		return []
	validator = Draft202012Validator(schema)
	errors = []
	for err in sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path)):
		where = "/".join(str(p) for p in err.absolute_path)
		errors.append(f"{where or '(root)'}: {err.message}")
		if len(errors) >= 5:
			break
	return errors


def _kind(value):
	if value is None:
		return "null"
	if isinstance(value, bool):
		return "boolean"
	if isinstance(value, int):
		return "integer"
	if isinstance(value, float):
		return "number"
	if isinstance(value, str):
		return "string"
	if isinstance(value, list):
		return "array"
	if isinstance(value, dict):
		return "object"
	return "string"


def infer_schema(values, enum_threshold=12):
	"""Build a JSON Schema that every example output satisfies.

	Short strings that repeat across examples become enums, which is what makes a
	classifier's label set explicit to the model and to constrained decoding.
	"""
	values = list(values)
	if not values:
		return {"type": "object", "properties": {}, "additionalProperties": False}
	return _infer(values, enum_threshold)


def _infer(values, threshold):
	groups = {}
	for v in values:
		groups.setdefault(_kind(v), []).append(v)
	if "integer" in groups and "number" in groups:
		groups["number"] = groups.pop("integer") + groups["number"]

	schemas = []
	for kind, vs in groups.items():
		if kind == "object":
			keys = list(dict.fromkeys(k for v in vs for k in v))
			props = {k: _infer([v[k] for v in vs if k in v], threshold) for k in keys}
			required = [k for k in keys if all(k in v for v in vs)]
			schemas.append({
				"type": "object",
				"properties": props,
				"required": required,
				"additionalProperties": False,
			})
		elif kind == "array":
			items = [i for v in vs for i in v]
			s = {"type": "array"}
			if items:
				s["items"] = _infer(items, threshold)
			schemas.append(s)
		elif kind == "string":
			distinct = list(dict.fromkeys(vs))
			if (
				len(distinct) <= threshold
				and len(vs) > len(distinct)
				and all(len(s) <= 48 and "\n" not in s for s in distinct)
			):
				schemas.append({"type": "string", "enum": distinct})
			else:
				schemas.append({"type": "string"})
		else:
			schemas.append({"type": kind})

	if len(schemas) == 1:
		return schemas[0]
	non_null = [s for s in schemas if s.get("type") != "null"]
	if len(non_null) == 1 and len(schemas) == 2:
		merged = dict(non_null[0])
		merged["type"] = [merged["type"], "null"]
		if "enum" in merged:
			merged["enum"] = merged["enum"] + [None]
		return merged
	return {"anyOf": schemas}


def _types(prop_schema):
	t = (prop_schema or {}).get("type")
	if isinstance(t, list):
		return t
	return [t] if t else []


def coerce_value(raw, prop_schema):
	"""Convert a CSV cell (always text) into the type the schema expects."""
	if raw is None:
		return None
	text = str(raw).strip()
	types = _types(prop_schema)
	if text == "" and "null" in types:
		return None
	if "integer" in types:
		try:
			return int(text)
		except ValueError:
			pass
	if "number" in types:
		try:
			return float(text)
		except ValueError:
			pass
	if "boolean" in types and text.lower() in ("true", "false", "yes", "no", "1", "0"):
		return text.lower() in ("true", "yes", "1")
	if "array" in types or "object" in types:
		try:
			return json.loads(text)
		except ValueError:
			if "array" in types:
				return [p.strip() for p in text.replace(";", "|").split("|") if p.strip()]
	return str(raw)


TEMPLATES = [
	{
		"id": "classification",
		"name": "Classification (one label)",
		"instructions": "Classify the text into exactly one category.",
		"schema": {
			"type": "object",
			"properties": {
				"label": {"type": "string", "enum": ["billing", "technical", "account", "other"]},
			},
			"required": ["label"],
			"additionalProperties": False,
		},
	},
	{
		"id": "multilabel",
		"name": "Tagging (multiple labels)",
		"instructions": "Assign every tag that applies to the text.",
		"schema": {
			"type": "object",
			"properties": {
				"tags": {
					"type": "array",
					"items": {"type": "string", "enum": ["urgent", "complaint", "question", "feedback"]},
				},
			},
			"required": ["tags"],
			"additionalProperties": False,
		},
	},
	{
		"id": "extraction",
		"name": "Field extraction",
		"instructions": "Extract the requested fields from the document. Use null when a field is not present.",
		"schema": {
			"type": "object",
			"properties": {
				"vendor": {"type": ["string", "null"]},
				"invoice_number": {"type": ["string", "null"]},
				"date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
				"total": {"type": ["number", "null"]},
			},
			"required": ["vendor", "invoice_number", "date", "total"],
			"additionalProperties": False,
		},
	},
	{
		"id": "file-analysis",
		"name": "File analysis report",
		"instructions": "Analyze the file contents and report its type, a short summary and any problems found.",
		"schema": {
			"type": "object",
			"properties": {
				"file_type": {"type": "string", "enum": ["config", "log", "source", "data", "document", "unknown"]},
				"summary": {"type": "string"},
				"issues": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"severity": {"type": "string", "enum": ["info", "warning", "error"]},
							"description": {"type": "string"},
						},
						"required": ["severity", "description"],
						"additionalProperties": False,
					},
				},
			},
			"required": ["file_type", "summary", "issues"],
			"additionalProperties": False,
		},
	},
	{
		"id": "sentiment",
		"name": "Sentiment with score",
		"instructions": "Rate the sentiment of the text.",
		"schema": {
			"type": "object",
			"properties": {
				"sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
				"score": {"type": "integer", "minimum": 1, "maximum": 5},
			},
			"required": ["sentiment", "score"],
			"additionalProperties": False,
		},
	},
]


def get_template(template_id):
	for t in TEMPLATES:
		if t["id"] == template_id:
			return t
	return TEMPLATES[0]
