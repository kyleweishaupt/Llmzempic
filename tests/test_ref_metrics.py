import json

import pytest

from llmzempic.datasets import Library
from llmzempic.ref_metrics import ref_scores
from llmzempic.refs import Resolver

SCHEMA = {
	"type": "object",
	"properties": {
		"a": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}, "maxItems": 1},
		"b": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}},
	},
}


def item(expected, generated):
	return {"expected": expected, "generated": json.dumps(generated) if not isinstance(generated, str) else generated}


@pytest.fixture
def resolver(tmp_path):
	source = tmp_path / "codes.txt"
	source.write_text("\n".join(["A", "B", "C", "D"] + [f"V{i}" for i in range(600)]) + "\n", encoding="utf-8")
	library = Library(tmp_path / "datasets")
	library.create("Codes", source, "codes.txt")
	return library.resolver()


def field(scores, name):
	return next(s for s in scores if s["field"] == name)


def test_partial_credit_math(resolver):
	scores = ref_scores([item({"a": ["A"], "b": ["A", "B", "C"]}, {"a": ["A"], "b": ["A", "C", "D"]})], SCHEMA, resolver)
	b = field(scores, "b")
	assert b["name"] == "Codes" and b["count"] == 1
	assert b["precision"] == pytest.approx(2 / 3, abs=1e-4) and b["recall"] == pytest.approx(2 / 3, abs=1e-4)
	assert b["f1"] == pytest.approx(2 / 3, abs=1e-4)
	assert b["exact"] == 0 and b["first"] == 1
	assert field(scores, "a")["exact"] == 1


def test_order_matters_for_exact(resolver):
	scores = ref_scores([item({"b": ["A", "B"]}, {"b": ["B", "A"]}), item({"b": ["A", "B"]}, {"b": ["A", "B"]})], SCHEMA, resolver)
	b = field(scores, "b")
	assert b["exact"] == 0.5 and b["precision"] == 1 and b["first"] == 0.5


def test_unparseable_answers_count_as_misses(resolver):
	b = field(ref_scores([item({"b": ["A"]}, "not json")], SCHEMA, resolver), "b")
	assert (b["exact"], b["recall"], b["first"]) == (0, 0, 0) and b["precision"] is None


def test_invalid_rate_uses_resolver(resolver):
	items = [item({"b": ["A"]}, {"b": ["A", "ZZ"]}), item({"b": ["B"]}, {"b": ["B"]})]
	assert field(ref_scores(items, SCHEMA, resolver), "b")["invalid_rate"] == 0.5
	assert field(ref_scores(items, SCHEMA, Resolver({})), "b")["invalid_rate"] == 1


def test_value_table_is_bounded(resolver):
	items = [item({"b": [f"V{i}"]}, {"b": [f"V{i}" if i % 3 else "A"]}) for i in range(500)]
	items += [item({"b": ["V7"]}, {"b": ["A"]}) for _ in range(5)]
	values = field(ref_scores(items, SCHEMA, resolver), "b")["values"]
	assert len(values) <= 30
	v7 = next(v for v in values if v["value"] == "V7")
	assert (v7["expected"], v7["correct"]) == (6, 1)
	assert values[0]["value"] == "V7"  # the most frequent expected value leads
