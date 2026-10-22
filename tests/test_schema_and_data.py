from llmzempic.data import parse_examples
from llmzempic.metrics import parse_generated, score_predictions
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
