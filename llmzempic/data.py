"""Importing and normalizing fine-tuning examples ({"input": text, "output": structured value})."""

import csv
import io
import json
import uuid

from .images import IMAGE_NAME
from .schema_tools import coerce_value

# Inputs can be whole documents; the default limit (131,072 characters) is far too small. 2**31 - 1
# because the limit is a C long, which is 32 bits on Windows.
csv.field_size_limit(2**31 - 1)

INPUT_KEYS = ("input", "prompt", "text", "question", "document", "content", "instruction")
OUTPUT_KEYS = ("output", "completion", "response", "answer", "expected", "target")


class InvalidExample(ValueError):
	"""An example that can't be stored as given."""


def image_name(value):
	# Only names of uploaded images: anything else could point at any file on disk.
	name = str(value)
	if not IMAGE_NAME.match(name):
		raise InvalidExample(f"image {name[:60]!r} isn't the name of an uploaded image")
	return name


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
	row = {
		"id": str(example.get("id") or new_id()),
		"input": "" if example.get("input") is None else str(example.get("input")),
		"output": parse_output(example.get("output")),
	}
	if example.get("image"):
		row["image"] = image_name(example["image"])
	return row


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
	image_key = _pick(record, ("image",))
	if input_key is None and image_key is None:
		raise ValueError(f"no input field (expected one of: {', '.join(INPUT_KEYS)})")
	output_key = _pick(record, OUTPUT_KEYS)
	if output_key is not None:
		output = parse_output(record[output_key])
	else:
		# Remaining columns become the output object, typed by the schema (handy for CSV label columns).
		props = (schema or {}).get("properties") or {}
		output = {}
		for key, value in record.items():
			if key in (input_key, image_key, "id"):
				continue
			output[key] = coerce_value(value, props.get(key)) if isinstance(value, str) else value
		if not output:
			raise ValueError("no output field or output columns")
	return {
		"input": record[input_key] if input_key else "",
		"output": output,
		"image": image_name(record[image_key]) if image_key and record[image_key] else None,
	}


def _raw_records(f, name):
	"""Yield (line number, parsed record or None, error or None) from an open text file."""
	if name.endswith((".csv", ".tsv")):
		reader = csv.DictReader(f, dialect="excel-tab" if name.endswith(".tsv") else "excel")
		for record in reader:
			yield reader.line_num, dict(record), None
		return
	start = f.read(4096)
	f.seek(0)
	if start.lstrip("\ufeff \t\r\n").startswith("["):
		try:
			records = json.load(f)
		except ValueError as e:
			yield e.lineno, None, f"not valid JSON ({e.msg})"
			return
		for i, record in enumerate(records, start=1):
			yield i, record, None
		return
	for i, line in enumerate(f, start=1):
		line = line.lstrip("\ufeff").strip()
		if not line:
			continue
		try:
			yield i, json.loads(line), None
		except ValueError as e:
			yield i, None, f"not valid JSON ({e.msg})"


def _iter_file(f, name, schema):
	for where, record, error in _raw_records(f, name.lower()):
		if error is None and not isinstance(record, dict):
			error = "expected an object"
		if error is None:
			try:
				example = _from_record(record, schema)
				record_id = record.get(_pick(record, ("id",)) or "id")
				if record_id not in (None, ""):
					example["id"] = str(record_id)
				yield where, example, None
				continue
			except ValueError as e:
				error = str(e)
		yield where, None, f"Line {where}: {error}"


def iter_records(path, filename, schema=None):
	"""Stream examples from a JSONL, JSON array or CSV/TSV file. Yields (line number, example or
	None, warning or None); a bad line becomes a warning and never stops the import."""
	with open(path, "r", encoding="utf-8-sig", newline="") as f:
		yield from _iter_file(f, filename or "", schema)


def parse_examples(text, filename="", schema=None):
	"""Parse JSONL, a JSON array, or CSV/TSV text into examples. Returns (examples, warnings)."""
	examples, warnings = [], []
	for _, example, warning in _iter_file(io.StringIO(text.lstrip("\ufeff"), newline=""), filename or "", schema):
		if example is not None:
			examples.append(normalize_example(example))
		else:
			warnings.append(warning)
	return examples, warnings
