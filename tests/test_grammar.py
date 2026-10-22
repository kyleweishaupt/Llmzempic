import re

import pytest

from llmzempic.grammar import values_grammar

CODES = {"type": "string", "x-dataset": "codes"}


def rules(text):
	return dict(line.split(" ::= ", 1) for line in text.splitlines() if " ::= " in line)


def references(body):
	# Identifiers left once string literals, character classes and repetition counts are removed.
	body = re.sub(r'"(?:[^"\\]|\\.)*"', " ", body)
	body = re.sub(r"\[(?:[^\]\\]|\\.)*\]", " ", body)
	body = re.sub(r"\{[\d,]*\}", " ", body)
	return set(re.findall(r"[a-zA-Z0-9-]+", body))


def test_values_share_prefixes():
	text = values_grammar(CODES, lambda _: ["A00", "A01", "B1"])
	assert '"A0" ' in text
	assert '"0\\"" | "1\\""' in text
	assert '"B1\\""' in text


def test_values_are_matched_as_json_text():
	text = values_grammar(CODES, lambda _: ['say "hi"', "back\\slash", "café"])
	assert '"back\\\\\\\\slash\\""' in text  # JSON doubles the backslash, GBNF doubles both again
	assert '"say \\\\\\"hi\\\\\\"\\""' in text
	assert '"café\\""' in text


def test_every_referenced_rule_is_defined():
	schema = {
		"type": "object",
		"properties": {
			"primary": {"type": "array", "items": CODES, "minItems": 1, "maxItems": 1},
			"supporting": {"type": "array", "items": CODES, "maxItems": 5},
			"label": {"type": "string", "enum": ["a", "b"]},
			"score": {"type": ["number", "null"]},
			"note": {"type": "string", "description": "optional", "pattern": "^x"},
			"tags": {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}, "uniqueItems": True},
			"meta": {"type": "object", "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}}, "required": ["ok"]},
			"anything": {},
		},
		"required": ["primary", "supporting", "label"],
		"additionalProperties": False,
	}
	text = values_grammar(schema, lambda _: ["A00", "B1"])
	defined = rules(text)
	assert "root" in defined
	for name, body in defined.items():
		assert references(body) <= set(defined), f"{name} refers to an undefined rule"


def test_properties_keep_schema_order():
	schema = {
		"type": "object",
		"properties": {"b": {"type": "string"}, "a": {"type": "integer"}},
		"required": ["b", "a"],
		"additionalProperties": False,
	}
	defined = rules(values_grammar(schema, lambda _: []))
	kv = {name: key for key in "ba" for name, body in defined.items() if body.startswith(f'"\\"{key}\\""')}
	assert [kv[n] for n in re.findall(r"[a-zA-Z0-9-]+", defined["root"]) if n in kv] == ["b", "a"]


def test_list_sizes_become_repetition_counts():
	text = values_grammar({"type": "array", "items": CODES, "minItems": 2, "maxItems": 4}, lambda _: ["A"])
	assert "{1,3}" in rules(text)["root"]


def test_empty_data_set_is_an_error():
	with pytest.raises(ValueError, match="codes"):
		values_grammar(CODES, lambda _: [])
