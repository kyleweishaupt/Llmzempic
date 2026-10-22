"""Label fields, input identity and the train/held-out split. Dataset statistics live in checks.py."""

import hashlib
import json
import random
from collections import Counter

from .refs import ref_values
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


def input_key(example):
	"""Examples with the same image and text (ignoring whitespace) count as the same input. A digest, so
	stores and splits don't hold every long input a second time; empty when there's no input at all."""
	text = " ".join(str(example.get("input") or "").split())
	image = example.get("image")
	key = f"{image}\n{text}" if image else text
	return hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest() if key else ""


def _stratify_field(schema):
	if check_schema(schema):
		return None
	return next((name for name, kind, _ in label_fields(schema) if kind != "tags"), None)


def split_keys(rows, schema):
	"""The little the split needs per example: id, input group, label stratum and data set values."""
	field = _stratify_field(schema)
	refs = not check_schema(schema)
	keys = []
	for row in rows:
		output = row.get("output")
		stratum = value_key(output[field]) if field and isinstance(output, dict) and field in output else None
		values = frozenset(f"{loc.split('/')[0]}\x1f{value}" for loc, _, value in ref_values(output, schema)) if refs else frozenset()
		keys.append({"id": row["id"], "group": input_key(row) or f"\x00{row['id']}", "stratum": stratum, "values": values})
	return keys


def _ranked(groups, rng):
	"""Groups ordered so taking from the front spreads evenly over every label stratum."""
	strata = {}
	for group in groups:
		strata.setdefault(group[0]["stratum"], []).append(group)
	ranked = []
	for members in strata.values():
		rng.shuffle(members)
		ranked += [((i + rng.random()) / len(members), group) for i, group in enumerate(members)]
	ranked.sort(key=lambda r: r[0])
	return [group for _, group in ranked]


def _uses(group):
	"""How many examples in a group carry each data set value."""
	counts = Counter()
	for key in group:
		counts.update(key["values"])
	return counts


def _can_remove(group, support):
	return all(support[v] - n >= 1 for v, n in _uses(group).items())


def split_examples(keys, schema, fraction, seed, held_out_max=300, max_examples=0):
	"""Pick the examples to train on and to hold out. Returns (train_ids, eval_ids, info).

	Copies of the same input stay on one side, so the score isn't inflated by examples the model
	trained on. Groups are spread evenly over the values of the first enum or boolean field. A data
	set value is never sampled away or held out when that would leave training without it, so the
	model sees every value it has examples for.
	"""
	rng = random.Random(seed)
	groups = {}
	for key in keys:
		groups.setdefault(key["group"], []).append(key)
	support = Counter()
	for group in groups.values():
		support.update(_uses(group))

	n = len(keys)
	target = min(n, max_examples) if max_examples > 0 else n
	protected = sum(len(g) for g in groups.values() if not _can_remove(g, support))

	# Hold out first, so sampling can't use up every example that could be scored.
	n_eval = min(held_out_max, int(round(target * fraction))) if target >= 10 else 0
	eval_groups, held = [], 0
	candidates = [g for g in _ranked(list(groups.values()), rng) if _can_remove(g, support)]
	for group in candidates:
		if held + len(group) <= n_eval and held + len(group) < n and _can_remove(group, support):
			eval_groups.append(group)
			support.subtract(_uses(group))
			held += len(group)
	if n_eval > 0 and not eval_groups and candidates:
		# Every input has more copies than the held-out size: hold out the smallest group anyway,
		# as long as training keeps at least one row.
		smallest = min(candidates, key=len)
		if len(smallest) < n:
			eval_groups.append(smallest)
			support.subtract(_uses(smallest))
	eval_set = {id(g) for g in eval_groups}
	train_groups = [g for g in groups.values() if id(g) not in eval_set]

	if n > target:
		dropped = set()
		for group in reversed(_ranked(train_groups, rng)):
			if n <= target:
				break
			if _can_remove(group, support):
				support.subtract(_uses(group))
				dropped.add(id(group))
				n -= len(group)
		train_groups = [g for g in train_groups if id(g) not in dropped]

	train_ids = [k["id"] for g in train_groups for k in g]
	eval_ids = [k["id"] for g in eval_groups for k in g]
	# The trainer may score the first examples only, so don't leave them ordered by label.
	rng.shuffle(train_ids)
	rng.shuffle(eval_ids)
	info = {
		"stratified_by": _stratify_field(schema), "groups": len(groups),
		"protected": protected, "sampled_out": len(keys) - n,
	}
	return train_ids, eval_ids, info
