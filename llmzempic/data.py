"""Importing and normalizing fine-tuning examples ({"input": text, "output": structured value})."""

import csv
import io
import json
import uuid

from .schema_tools import coerce_value

INPUT_KEYS = ("input", "prompt", "text", "question", "document", "content", "instruction")
OUTPUT_KEYS = ("output", "completion", "response", "answer", "expected", "target")


def new_id():
	return uuid.uuid4().hex[:10]


def parse_output(value):
	"""Expected outputs may arrive as JSON text; decode them so they can be validated."""
	if isinstance(value, str):
		text = value.strip()
		if text[:1] in ("{", "[", '"') or text in ("true", "false", "null"):
			try:
				return json.loads(text)
			except ValueError:
				pass
	return value


def normalize_example(example):
	return {
		"id": str(example.get("id") or new_id()),
		"input": "" if example.get("input") is None else str(example.get("input")),
		"output": parse_output(example.get("output")),
	}


def _pick(record, keys):
	lowered = {k.lower(): k for k in record}
	for key in keys:
		if key in lowered:
			return lowered[key]
	return None


def _from_record(record, schema):
	if isinstance(record.get("messages"), list):
		user = [m.get("content", "") for m in record["messages"] if m.get("role") == "user"]
		assistant = [m.get("content", "") for m in record["messages"] if m.get("role") == "assistant"]
		if not user or not assistant:
			raise ValueError("messages need a user and an assistant turn")
		return {"input": user[-1], "output": parse_output(assistant[-1])}

	input_key = _pick(record, INPUT_KEYS)
	if input_key is None:
		raise ValueError(f"no input field (expected one of: {', '.join(INPUT_KEYS)})")
	output_key = _pick(record, OUTPUT_KEYS)
	if output_key is not None:
		output = parse_output(record[output_key])
	else:
		# Remaining columns become the output object, typed by the schema (handy for CSV label columns).
		props = (schema or {}).get("properties") or {}
		output = {}
		for key, value in record.items():
			if key == input_key or key == "id":
				continue
			output[key] = coerce_value(value, props.get(key)) if isinstance(value, str) else value
		if not output:
			raise ValueError("no output field or output columns")
	return {"input": record[input_key], "output": output}


def parse_examples(text, filename="", schema=None):
	"""Parse JSONL, a JSON array, or CSV/TSV text into examples. Returns (examples, warnings)."""
	text = text.lstrip("﻿")
	name = (filename or "").lower()
	warnings = []
	records = []

	if name.endswith((".csv", ".tsv")):
		dialect = "excel-tab" if name.endswith(".tsv") else "excel"
		reader = csv.DictReader(io.StringIO(text), dialect=dialect)
		records = [(i + 2, dict(r)) for i, r in enumerate(reader)]
	elif text.strip().startswith("["):
		data = json.loads(text)
		records = [(i + 1, r) for i, r in enumerate(data)]
	else:
		for i, line in enumerate(text.splitlines(), start=1):
			if not line.strip():
				continue
			try:
				records.append((i, json.loads(line)))
			except ValueError as e:
				warnings.append(f"Line {i}: not valid JSON ({e.msg})")

	examples = []
	for where, record in records:
		if not isinstance(record, dict):
			warnings.append(f"Record {where}: expected an object")
			continue
		try:
			examples.append(normalize_example(_from_record(record, schema)))
		except ValueError as e:
			warnings.append(f"Record {where}: {e}")
	return examples, warnings
