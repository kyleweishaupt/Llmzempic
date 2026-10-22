from llmzempic.dataset import dataset_stats, split_examples
from llmzempic.schema_tools import TEMPLATES

JUNK_OUTPUTS = [None, "not json", 7, [], ["a"], {"label": None}, {"tags": "urgent"}, {"tags": [{"x": 1}, None]}, {"is_spam": 1}]


def counts(stats, field):
	entry = next(f for f in stats["fields"] if f["field"] == field)
	return {f"{v['value']}": v["count"] for v in entry["values"]}


def test_every_template_handles_junk_outputs():
	examples = [{"id": str(i), "input": f"text {i}", "output": out} for i, out in enumerate(JUNK_OUTPUTS)]
	for template in TEMPLATES:
		stats = dataset_stats(examples, template["schema"])
		assert stats["count"] == len(examples), template["id"]
		assert stats["duplicates"] == []
		for field in stats["fields"]:
			assert all(v["count"] == 0 for v in field["values"]), (template["id"], field)


def test_label_fields_by_kind():
	stats = dataset_stats([], next(t for t in TEMPLATES if t["id"] == "file-analysis")["schema"])
	assert [(f["field"], f["kind"]) for f in stats["fields"]] == [("file_type", "enum")]
	stats = dataset_stats([], next(t for t in TEMPLATES if t["id"] == "extraction")["schema"])
	assert stats["fields"] == []


def test_enum_counts_include_unseen_labels():
	schema = next(t for t in TEMPLATES if t["id"] == "classification")["schema"]
	examples = [
		{"id": "1", "input": "a", "output": {"label": "billing"}},
		{"id": "2", "input": "b", "output": {"label": "billing"}},
		{"id": "3", "input": "c", "output": {"label": "technical"}},
		{"id": "4", "input": "d", "output": {"label": "made-up"}},
	]
	assert counts(dataset_stats(examples, schema), "label") == {"billing": 2, "technical": 1, "account": 0, "other": 0}


def test_tags_counted_once_per_example():
	schema = next(t for t in TEMPLATES if t["id"] == "multilabel")["schema"]
	examples = [
		{"id": "1", "input": "a", "output": {"tags": ["urgent", "urgent", "question"]}},
		{"id": "2", "input": "b", "output": {"tags": ["urgent"]}},
	]
	stats = dataset_stats(examples, schema)
	assert stats["fields"][0]["kind"] == "tags"
	assert counts(stats, "tags") == {"urgent": 2, "complaint": 0, "question": 1, "feedback": 0}


def test_booleans_and_nullable_enums_keep_types_apart():
	schema = {
		"type": "object",
		"properties": {
			"is_spam": {"type": "boolean"},
			"level": {"type": ["integer", "null"], "enum": [1, 2, None]},
		},
	}
	examples = [
		{"id": "1", "input": "a", "output": {"is_spam": True, "level": 1}},
		{"id": "2", "input": "b", "output": {"is_spam": 1, "level": True}},
		{"id": "3", "input": "c", "output": {"is_spam": False, "level": None}},
	]
	stats = dataset_stats(examples, schema)
	assert counts(stats, "is_spam") == {"True": 1, "False": 1}
	assert counts(stats, "level") == {"1": 1, "2": 0, "None": 1}


def test_invalid_schema_still_reports_duplicates():
	examples = [{"id": "1", "input": "same", "output": 1}, {"id": "2", "input": "same", "output": 1}]
	stats = dataset_stats(examples, {"type": "nonsense"})
	assert stats["fields"] == []
	assert stats["duplicates"] == [{"ids": ["1", "2"], "conflict": False}]


def test_duplicates_normalize_whitespace_and_flag_conflicts():
	examples = [
		{"id": "1", "input": "Refund  my\norder", "output": {"label": "billing", "x": 1}},
		{"id": "2", "input": " Refund my order ", "output": {"x": 1, "label": "billing"}},
		{"id": "3", "input": "Login fails", "output": {"label": "technical"}},
		{"id": "4", "input": "Login fails", "output": {"label": "account"}},
		{"id": "5", "input": "login fails", "output": {"label": "account"}},
		{"id": "6", "input": "  ", "output": {"label": "other"}},
		{"id": "7", "input": "", "output": {"label": "other"}},
	]
	stats = dataset_stats(examples, {"type": "object"})
	assert stats["duplicates"] == [
		{"ids": ["1", "2"], "conflict": False},
		{"ids": ["3", "4"], "conflict": True},
	]


CLASSIFY = next(t for t in TEMPLATES if t["id"] == "classification")["schema"]


def labeled(n_billing, n_technical):
	labels = ["billing"] * n_billing + ["technical"] * n_technical
	return [{"id": str(i), "input": f"ticket {i}", "output": {"label": label}} for i, label in enumerate(labels)]


def ids(rows):
	return sorted(r["id"] for r in rows)


def test_split_is_stratified_by_label():
	for seed in range(5):
		train, held_out, info = split_examples(labeled(70, 30), CLASSIFY, 0.2, seed)
		assert info == {"stratified_by": "label", "groups": 100}
		assert len(held_out) == 20 and len(train) == 80
		counts = {label: sum(r["output"]["label"] == label for r in held_out) for label in ("billing", "technical")}
		assert 13 <= counts["billing"] <= 15 and 5 <= counts["technical"] <= 7, counts


def test_split_keeps_copies_together():
	examples = labeled(20, 20)
	for i in range(0, 12, 3):
		examples.append({"id": f"copy{i}", "input": f"  ticket\n{i} ", "output": examples[i]["output"]})
		examples.append({"id": f"conflict{i}", "input": f"ticket {i}", "output": {"label": "other"}})
	for seed in range(10):
		train, held_out, info = split_examples(examples, CLASSIFY, 0.25, seed)
		assert info["groups"] == 40
		assert len(train) + len(held_out) == len(examples)
		train_inputs = {" ".join(r["input"].split()) for r in train}
		assert not any(" ".join(r["input"].split()) in train_inputs for r in held_out)


def test_split_is_deterministic_per_seed():
	examples = labeled(30, 30)
	first = split_examples(examples, CLASSIFY, 0.2, 7)
	assert [r["id"] for r in first[1]] == [r["id"] for r in split_examples(examples, CLASSIFY, 0.2, 7)[1]]
	assert ids(first[1]) != ids(split_examples(examples, CLASSIFY, 0.2, 8)[1])


def test_split_edge_cases():
	assert split_examples(labeled(5, 4), CLASSIFY, 0.5, 1)[1] == []
	assert split_examples(labeled(10, 10), CLASSIFY, 0, 1)[1] == []
	train, held_out, _ = split_examples(labeled(10, 10), CLASSIFY, 1.0, 1)
	assert len(train) >= 1 and len(train) + len(held_out) == 20
	extraction = next(t for t in TEMPLATES if t["id"] == "extraction")["schema"]
	assert split_examples(labeled(10, 10), extraction, 0.1, 1)[2]["stratified_by"] is None
	assert len(split_examples(labeled(10, 10), {"type": "nonsense"}, 0.1, 1)[1]) == 2
	pairs = [{"id": f"{i}{c}", "input": f"t{i}", "output": {"label": "billing"}} for i in range(6) for c in "ab"]
	held_out = split_examples(pairs, CLASSIFY, 0.1, 1)[1]
	assert len(held_out) == 2 and held_out[0]["input"] == held_out[1]["input"]
	big = labeled(20, 20) + [{"id": f"c{i}", "input": "same", "output": {"label": "billing"}} for i in range(10)]
	for seed in range(20):
		assert len(split_examples(big, CLASSIFY, 0.1, seed)[1]) == 5

