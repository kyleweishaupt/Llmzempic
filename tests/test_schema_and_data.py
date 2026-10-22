import json

from llmzempic.data import parse_examples
from llmzempic.metrics import label_breakdown, parse_generated, score_predictions
from llmzempic.schema_tools import TEMPLATES, check_schema, coerce_value, infer_schema, validate_output

LABEL_SCHEMA = {
	"type": "object",
	"properties": {
		"label": {"type": "string", "enum": ["a", "b"]},
		"score": {"type": "integer"},
	},
	"required": ["label"],
	"additionalProperties": False,
}


def test_templates_are_valid_schemas():
	for template in TEMPLATES:
		assert check_schema(template["schema"]) == [], template["id"]


def test_check_schema_reports_problems():
	assert check_schema([]) != []
	assert check_schema({"type": "nonsense"}) != []
	assert check_schema({}) != []


def test_validate_output():
	assert validate_output({"label": "a"}, LABEL_SCHEMA) == []
	errors = validate_output({"label": "c", "extra": 1}, LABEL_SCHEMA)
	assert any("label" in e for e in errors)
	assert len(errors) == 2


def test_infer_schema_builds_enums_and_required():
	outputs = [
		{"label": "spam", "score": 1},
		{"label": "ham", "score": 2},
		{"label": "spam", "score": 3.5, "note": None},
		{"label": "ham", "score": 4, "note": "x"},
	]
	schema = infer_schema(outputs)
	assert schema["properties"]["label"] == {"type": "string", "enum": ["spam", "ham"]}
	assert schema["properties"]["score"] == {"type": "number"}
	assert schema["properties"]["note"]["type"] == ["string", "null"]
	assert schema["required"] == ["label", "score"]
	for output in outputs:
		assert validate_output(output, schema) == []


def test_coerce_value_uses_schema_types():
	assert coerce_value("3", {"type": "integer"}) == 3
	assert coerce_value("2.5", {"type": "number"}) == 2.5
	assert coerce_value("yes", {"type": "boolean"}) is True
	assert coerce_value("", {"type": ["string", "null"]}) is None
	assert coerce_value("a|b", {"type": "array"}) == ["a", "b"]
	assert coerce_value("hello", None) == "hello"


def test_parse_jsonl_and_messages():
	text = "\n".join([
		'{"input": "one", "output": {"label": "a"}}',
		'{"prompt": "two", "completion": "{\\"label\\": \\"b\\"}"}',
		'{"messages": [{"role": "user", "content": "three"}, {"role": "assistant", "content": "{\\"label\\": \\"a\\"}"}]}',
		"not json",
		'{"nothing": 1}',
	])
	examples, warnings = parse_examples(text, "data.jsonl", LABEL_SCHEMA)
	assert [e["input"] for e in examples] == ["one", "two", "three"]
	assert examples[1]["output"] == {"label": "b"}
	assert all(e["id"] for e in examples)
	assert len(warnings) == 2


def test_parse_csv_columns_become_typed_fields():
	text = "text,label,score\nhello,a,3\nbye,b,\n"
	examples, warnings = parse_examples(text, "data.csv", LABEL_SCHEMA)
	assert warnings == []
	assert examples[0] == {"id": examples[0]["id"], "input": "hello", "output": {"label": "a", "score": 3}}
	assert examples[1]["output"]["label"] == "b"


def test_parse_json_array():
	examples, _ = parse_examples('[{"input": "x", "output": {"label": "a"}}]', "data.json")
	assert examples[0]["output"] == {"label": "a"}


def test_parse_generated_tolerates_fences_and_prefixes():
	assert parse_generated('{"a": 1}') == ({"a": 1}, True)
	assert parse_generated('```json\n{"a": 1}\n```') == ({"a": 1}, True)
	assert parse_generated('Sure! {"a": 1} done') == ({"a": 1}, True)
	assert parse_generated("nope")[1] is False


def test_score_predictions():
	rows = [
		{"input": "1", "output": {"label": "a"}},
		{"input": "2", "output": {"label": "b"}},
		{"input": "3", "output": {"label": "a"}},
	]
	result = score_predictions(rows, ['{"label":"a"}', '{"label":"a"}', "garbage"], LABEL_SCHEMA)
	assert result["count"] == 3
	assert result["json_valid"] == round(2 / 3, 4)
	assert result["schema_valid"] == round(2 / 3, 4)
	assert result["exact_match"] == round(1 / 3, 4)
	assert result["field_accuracy"] == {"label": round(1 / 3, 4)}


def test_label_breakdown_counts_recognized_predicted_and_mixups():
	rows = [{"input": str(i), "output": {"label": label}} for i, label in enumerate("aaabb")]
	generated = ['{"label":"a"}', '{"label":"a"}', '{"label":"b"}', '{"label":"a"}', "garbage"]
	items = score_predictions(rows, generated, LABEL_SCHEMA)["items"]
	[field] = label_breakdown(items, LABEL_SCHEMA)
	assert field["field"] == "label" and field["kind"] == "enum"
	assert field["values"] == [
		{"value": "a", "expected": 3, "predicted": 3, "correct": 2},
		{"value": "b", "expected": 2, "predicted": 1, "correct": 0},
	]
	assert field["confusions"] == [
		{"expected": "a", "predicted": "b", "unknown": False, "count": 1},
		{"expected": "b", "predicted": "a", "unknown": False, "count": 1},
		{"expected": "b", "predicted": None, "unknown": True, "count": 1},
	]


def test_label_breakdown_tags_booleans_and_bad_input():
	schema = {
		"type": "object",
		"properties": {
			"tags": {"type": "array", "items": {"enum": ["x", "y", "z"]}},
			"flag": {"type": "boolean"},
			"note": {"type": "string"},
		},
	}
	items = [
		{"expected": {"tags": ["x", "y"], "flag": True}, "generated": '{"tags":["x","z",{"o":1}],"flag":1}'},
		{"expected": {"tags": ["y"], "flag": False}, "generated": '{"tags":"y","flag":false}'},
		{"expected": "not an object", "generated": "{}"},
		{"expected": {"note": "only"}, "generated": None},
	]
	fields = {f["field"]: f for f in label_breakdown(items, schema)}
	assert fields["tags"]["values"] == [
		{"value": "x", "expected": 1, "predicted": 1, "correct": 1},
		{"value": "y", "expected": 2, "predicted": 0, "correct": 0},
		{"value": "z", "expected": 0, "predicted": 1, "correct": 0},
	]
	assert fields["tags"]["confusions"] == []
	assert fields["flag"]["values"] == [
		{"value": True, "expected": 1, "predicted": 0, "correct": 0},
		{"value": False, "expected": 1, "predicted": 1, "correct": 1},
	]
	assert fields["flag"]["confusions"] == [{"expected": True, "predicted": None, "unknown": True, "count": 1}]
	assert label_breakdown(items, {"type": "nonsense"}) == []
	assert label_breakdown([], schema) == []



def test_schema_is_checked_once_per_distinct_schema(monkeypatch):
	# The checker validates 200k outputs against one schema; re-checking the schema each time was 90% of it.
	from jsonschema import Draft202012Validator
	real = Draft202012Validator.check_schema
	calls = []
	monkeypatch.setattr(Draft202012Validator, "check_schema", lambda schema: calls.append(1) or real(schema))
	schema = {"title": "checked once", "type": "object", "properties": {"x": {"type": "integer"}}}
	for i in range(20):
		assert validate_output({"x": i}, dict(schema)) == []
	assert check_schema(dict(schema)) == []
	assert len(calls) == 1


def test_imported_image_must_name_an_uploaded_image(tmp_path):
	from llmzempic.data import iter_records
	path = tmp_path / "rows.jsonl"
	rows = [{"input": "", "image": "C:/Users/me/private-notes.txt", "output": {"x": 1}}, {"input": "", "image": "0123456789abcdef01234567.png", "output": {"x": 2}}]
	path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
	results = list(iter_records(path, "rows.jsonl"))
	assert results[0][1] is None and "image" in results[0][2]
	assert results[1][1]["image"] == "0123456789abcdef01234567.png"


def test_long_csv_cells_and_broken_json_arrays_never_crash_an_import(tmp_path):
	from llmzempic.data import iter_records
	long_text = "word " * 40000  # past Python's default 131,072-character CSV field limit
	csv_path = tmp_path / "rows.csv"
	csv_path.write_text(f'input,label\n"{long_text}",a\nshort,b\n', encoding="utf-8")
	assert [len(e["input"]) for _, e, _ in iter_records(csv_path, "rows.csv")] == [len(long_text), 5]
	json_path = tmp_path / "rows.json"
	json_path.write_text('[{"input": "a", "output": 1},\n{"input": ', encoding="utf-8")
	[(_, example, warning)] = list(iter_records(json_path, "rows.json"))
	assert example is None and warning.startswith("Line 2: not valid JSON")
