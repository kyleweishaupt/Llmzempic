"""Dataset health for the Examples tab (label coverage, repeated inputs) and the train/held-out split."""

import json
import random

from .schema_tools import check_schema


def _types(schema):
	kind = schema.get("type")
	return set(kind) if isinstance(kind, list) else {kind}


def value_key(value):
	"""Comparable form of a JSON value that keeps 1, true and "1" apart."""
	return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def label_fields(schema):
	"""Top-level fields with a fixed set of values: enums, booleans and arrays of enum tags."""
	fields = []
	for name, prop in (schema.get("properties") or {}).items():
		if not isinstance(prop, dict):
			continue
		items = prop.get("items")
		if isinstance(prop.get("enum"), list):
			fields.append((name, "enum", prop["enum"]))
		elif "boolean" in _types(prop):
			fields.append((name, "boolean", [True, False]))
		elif "array" in _types(prop) and isinstance(items, dict) and isinstance(items.get("enum"), list):
			fields.append((name, "tags", items["enum"]))
	return fields


def _field_counts(examples, name, kind, values):
	counts = {value_key(v): 0 for v in values}
	for example in examples:
		output = example.get("output")
		if not isinstance(output, dict) or name not in output:
			continue
		found = output[name]
		if kind == "tags":
			seen = {value_key(v) for v in found} if isinstance(found, list) else set()
		else:
			seen = {value_key(found)}
		for key in seen & counts.keys():
			counts[key] += 1
	return {"field": name, "kind": kind, "values": [{"value": v, "count": counts[value_key(v)]} for v in values]}


def normalize_input(text):
	"""Inputs that differ only in whitespace count as the same input."""
	return " ".join(str(text or "").split())


def _duplicates(examples):
	groups = {}
	for example in examples:
		text = normalize_input(example.get("input"))
		if text:
			groups.setdefault(text, []).append(example)
	return [
		{
			"ids": [e.get("id") for e in group],
			"conflict": len({value_key(e.get("output")) for e in group}) > 1,
		}
		for group in groups.values() if len(group) > 1
	]


def dataset_stats(examples, schema):
	"""Summarize examples for the UI. Never raises on malformed outputs or schemas."""
	fields = []
	if not check_schema(schema):
		fields = [_field_counts(examples, *field) for field in label_fields(schema)]
	return {"count": len(examples), "fields": fields, "duplicates": _duplicates(examples)}


def split_examples(examples, schema, fraction, seed):
	"""Pick the held-out examples. Returns (train_rows, eval_rows, info).

	Copies of the same input stay on one side, so the score isn't inflated by examples the
	model trained on. Groups are spread evenly over the values of the first enum or boolean
	field, so every label is held out in proportion to how often it occurs.
	"""
	rng = random.Random(seed)
	groups = {}
	for i, example in enumerate(examples):
		groups.setdefault(normalize_input(example.get("input")) or ("", i), []).append(example)
	field = None
	if not check_schema(schema):
		field = next((name for name, kind, _ in label_fields(schema) if kind != "tags"), None)
	strata = {}
	for group in groups.values():
		output = group[0].get("output")
		key = value_key(output[field]) if field and isinstance(output, dict) and field in output else None
		strata.setdefault(key, []).append(group)
	ranked = []
	for members in strata.values():
		rng.shuffle(members)
		ranked += [((i + rng.random()) / len(members), group) for i, group in enumerate(members)]
	ranked.sort(key=lambda r: r[0])

	n = len(examples)
	n_eval = min(200, int(round(n * fraction))) if n >= 10 else 0
	train_groups, eval_groups, held = [], [], 0
	for _, group in ranked:
		if held + len(group) <= n_eval and held + len(group) < n:
			eval_groups.append(group)
			held += len(group)
		else:
			train_groups.append(group)
	if n_eval > 0 and not eval_groups:
		# Every input has more copies than the held-out size: hold out the smallest group anyway,
		# as long as training keeps at least one row.
		smallest = min(train_groups, key=len)
		if len(smallest) < n:
			train_groups.remove(smallest)
			eval_groups.append(smallest)
	train_rows = [row for group in train_groups for row in group]
	eval_rows = [row for group in eval_groups for row in group]
	# The trainer scores the first examples only, so don't leave them ordered by label.
	rng.shuffle(train_rows)
	rng.shuffle(eval_rows)
	return train_rows, eval_rows, {"stratified_by": field, "groups": len(groups)}

