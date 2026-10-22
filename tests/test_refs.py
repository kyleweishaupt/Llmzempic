import json

import pytest

from llmzempic.datasets import Library
from llmzempic.prompts import build_system_prompt
from llmzempic.refs import Resolver, decoding_schema, prompt_schema, ref_errors, ref_fields, ref_values, referenced_ids, rewrite_ids
from llmzempic.schema_tools import validate_output

SECTIONS = {
	"type": "object",
	"properties": {
		"a": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}, "maxItems": 1},
		"b": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}},
	},
	"required": ["a", "b"],
	"additionalProperties": False,
}


@pytest.fixture
def resolver(tmp_path):
	source = tmp_path / "codes.txt"
	source.write_text("V1  First\nV2  Second\n", encoding="utf-8")
	library = Library(tmp_path / "datasets")
	library.create("Codes", source, "codes.txt")
	return library.resolver()


def test_ref_fields_finds_list_items():
	assert ref_fields(SECTIONS) == [
		{"path": ("a", "*"), "dataset": "codes", "list": True, "field": "a"},
		{"path": ("b", "*"), "dataset": "codes", "list": True, "field": "b"},
	]
	single = {"type": "object", "properties": {"c": {"type": "string", "x-dataset": "codes"}}}
	assert ref_fields(single) == [{"path": ("c",), "dataset": "codes", "list": False, "field": "c"}]


def test_ref_values_reports_locations():
	output = {"a": ["V1"], "b": ["V2", "V3"]}
	assert ref_values(output, SECTIONS) == [("a/0", "codes", "V1"), ("b/0", "codes", "V2"), ("b/1", "codes", "V3")]
	assert ref_values({"a": "not a list"}, SECTIONS) == []


def test_ref_errors_quote_exact_values(resolver):
	output = {"a": ["V1"], "b": ["v2", " V2", "V3"]}
	assert ref_errors(output, SECTIONS, resolver) == [
		'b/0: "v2" isn\'t in data set Codes',
		'b/1: " V2" isn\'t in data set Codes',
		'b/2: "V3" isn\'t in data set Codes',
	]


def test_unknown_dataset_rejects_every_value():
	empty = Resolver({})
	assert empty.missing("gone", ["V1"]) == {"V1"}
	assert empty.name("gone") == "gone"


def test_prompt_never_contains_values():
	prompt = build_system_prompt("Do it.", SECTIONS, True, {"codes": "Codes"})
	assert 'One value from the data set \\"Codes\\".' in prompt
	assert "x-dataset" not in prompt
	described = {"type": "object", "properties": {"c": {"type": "string", "x-dataset": "codes", "description": "Main one."}}}
	assert prompt_schema(described, {"codes": "Codes"})["properties"]["c"] == {
		"type": "string", "description": 'Main one. One value from the data set "Codes".',
	}


def test_decoding_schema_expands_enum():
	schema = decoding_schema(SECTIONS, lambda dataset_id: ["V1", "V2"])
	assert schema["properties"]["a"]["items"] == {"type": "string", "enum": ["V1", "V2"]}
	assert "x-dataset" in json.dumps(SECTIONS)


def test_rewrite_ids():
	rewritten = rewrite_ids(SECTIONS, {"codes": "codes-2"})
	assert referenced_ids(rewritten) == {"codes-2"}
	assert referenced_ids(SECTIONS) == {"codes"}


def test_validate_output_without_resolver_ignores_refs():
	assert validate_output({"a": ["X"], "b": []}, SECTIONS) == []


def test_validate_output_with_resolver(resolver):
	assert validate_output({"a": ["V1"], "b": ["V3"]}, SECTIONS, resolver) == ['b/0: "V3" isn\'t in data set Codes']
