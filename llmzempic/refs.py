"""Output fields whose values come from a reference data set: {"type": "string", "x-dataset": "<id>"},
on a field or on a list's items. JSON Schema validators ignore the keyword; these helpers find the
fields, check values against the data sets, and rewrite the keyword for prompts and decoding."""

import copy
import json

REF_KEY = "x-dataset"


def _walk(node, path=()):
	"""Yield (path, node) for every schema node holding the keyword. "*" stands for array items."""
	if not isinstance(node, dict):
		return
	if isinstance(node.get(REF_KEY), str):
		yield path, node
	for key, sub in (node.get("properties") or {}).items():
		yield from _walk(sub, path + (key,))
	if isinstance(node.get("items"), dict):
		yield from _walk(node["items"], path + ("*",))


def ref_fields(schema):
	return [
		{"path": path, "dataset": node[REF_KEY], "list": bool(path) and path[-1] == "*", "field": path[0] if path else ""}
		for path, node in _walk(schema)
	]


def _follow(value, path, location):
	if not path:
		yield location, value
		return
	step, rest = path[0], path[1:]
	if step == "*":
		if isinstance(value, list):
			for i, item in enumerate(value):
				yield from _follow(item, rest, location + (str(i),))
	elif isinstance(value, dict) and step in value:
		yield from _follow(value[step], rest, location + (step,))


def ref_values(output, schema):
	"""[(location, dataset id, value)] for every string an output holds in a data set field."""
	found = []
	for field in ref_fields(schema):
		for location, value in _follow(output, field["path"], ()):
			if isinstance(value, str):
				found.append(("/".join(location), field["dataset"], value))
	return found


class Resolver:
	"""Data set lookups for validation and scoring, over {id: DataSet}. Unknown ids accept nothing."""

	def __init__(self, datasets):
		self.datasets = dict(datasets)

	def missing(self, dataset_id, values):
		dataset = self.datasets.get(dataset_id)
		return dataset.missing(values) if dataset else set(values)

	def name(self, dataset_id):
		dataset = self.datasets.get(dataset_id)
		return (dataset.meta.get("name") if dataset else None) or dataset_id

	def hashes(self):
		return {dataset_id: d.meta.get("hash") for dataset_id, d in self.datasets.items()}


def ref_errors(output, schema, resolver):
	found = ref_values(output, schema)
	by_set = {}
	for _, dataset_id, value in found:
		by_set.setdefault(dataset_id, []).append(value)
	missing = {dataset_id: resolver.missing(dataset_id, values) for dataset_id, values in by_set.items()}
	return [
		f"{location}: {json.dumps(value, ensure_ascii=False)} isn't in data set {resolver.name(dataset_id)}"
		for location, dataset_id, value in found
		if value in missing[dataset_id]
	]


def _rewrite(schema, fn):
	schema = copy.deepcopy(schema)
	nodes = {id(node): node for _, node in _walk(schema)}  # a node can be shared by several fields
	for node in nodes.values():
		fn(node)
	return schema


def prompt_schema(schema, names):
	"""The schema the model reads: the keyword becomes a sentence, never the values."""
	def describe(node):
		dataset_id = node.pop(REF_KEY)
		sentence = f'One value from the data set "{names.get(dataset_id, dataset_id)}".'
		node["description"] = f"{node['description']} {sentence}" if node.get("description") else sentence
	return _rewrite(schema, describe)


def decoding_schema(schema, values_for):
	"""The schema a server constrains decoding with: the keyword becomes an enum of the values."""
	def expand(node):
		node["enum"] = list(values_for(node.pop(REF_KEY)))
	return _rewrite(schema, expand)


def export_contract(schema, datasets):
	"""What an export hands its clients: the schema they send (data sets named, values left out) and one
	entry per data set field naming the list file answers are checked against. `datasets` is a run's
	{id: {"name", ...}}."""
	names = {dataset_id: (meta or {}).get("name") or dataset_id for dataset_id, meta in (datasets or {}).items()}
	refs = [
		{"path": list(f["path"]), "file": f"datasets/{f['dataset']}.txt", "name": names.get(f["dataset"], f["dataset"])}
		for f in ref_fields(schema)
	]
	return (prompt_schema(schema, names) if refs else schema), refs


def referenced_ids(schema):
	return {node[REF_KEY] for _, node in _walk(schema)}


def rewrite_ids(schema, mapping):
	def swap(node):
		node[REF_KEY] = mapping.get(node[REF_KEY], node[REF_KEY])
	return _rewrite(schema, swap)
