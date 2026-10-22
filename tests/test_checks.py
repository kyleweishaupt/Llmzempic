import threading

import pytest

from llmzempic import checks
from llmzempic.checks import Checker, checks_key, stats
from llmzempic.example_store import ExampleStore
from llmzempic.schema_tools import TEMPLATES
from llmzempic.workspace import Workspace

JUNK_OUTPUTS = [None, "not json", 7, [], ["a"], {"label": None}, {"tags": "urgent"}, {"tags": [{"x": 1}, None]}, {"is_spam": 1}]
CODES = {
	"type": "object",
	"properties": {"b": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}}},
	"required": ["b"],
}


@pytest.fixture
def ws(tmp_path):
	return Workspace(tmp_path / "ws")


def project_with(ws, schema, examples, **config):
	project = ws.create("P")
	project.save_config({"schema": schema, **config})
	project.examples().import_records(examples, "append")
	return project


def run_stats(ws, project):
	return stats(project, ws.library, project.load_config())


def counts(result, field):
	entry = next(f for f in result["fields"] if f["field"] == field)
	return {f"{v['value']}": v["count"] for v in entry["values"]}


def template(template_id):
	return next(t for t in TEMPLATES if t["id"] == template_id)["schema"]


def test_every_template_handles_junk_outputs(ws):
	examples = [{"id": str(i), "input": f"text {i}", "output": out} for i, out in enumerate(JUNK_OUTPUTS)]
	project = project_with(ws, template("classification"), examples)
	for t in TEMPLATES:
		project.save_config({"schema": t["schema"]})
		result = run_stats(ws, project)
		assert result["count"] == len(examples), t["id"]
		assert result["duplicates"] == {"groups": 0, "conflicts": 0}
		for field in result["fields"]:
			assert all(v["count"] == 0 for v in field["values"]), (t["id"], field)


def test_label_fields_by_kind(ws):
	project = project_with(ws, template("file-analysis"), [])
	assert [(f["field"], f["kind"]) for f in run_stats(ws, project)["fields"]] == [("file_type", "enum")]
	project.save_config({"schema": template("extraction")})
	assert run_stats(ws, project)["fields"] == []


def test_enum_counts_include_unseen_labels(ws):
	examples = [
		{"id": "1", "input": "a", "output": {"label": "billing"}},
		{"id": "2", "input": "b", "output": {"label": "billing"}},
		{"id": "3", "input": "c", "output": {"label": "technical"}},
		{"id": "4", "input": "d", "output": {"label": "made-up"}},
	]
	project = project_with(ws, template("classification"), examples)
	assert counts(run_stats(ws, project), "label") == {"billing": 2, "technical": 1, "account": 0, "other": 0}


def test_tags_counted_once_per_example(ws):
	examples = [
		{"id": "1", "input": "a", "output": {"tags": ["urgent", "urgent", "question"]}},
		{"id": "2", "input": "b", "output": {"tags": ["urgent"]}},
	]
	result = run_stats(ws, project_with(ws, template("multilabel"), examples))
	assert result["fields"][0]["kind"] == "tags"
	assert counts(result, "tags") == {"urgent": 2, "complaint": 0, "question": 1, "feedback": 0}


def test_booleans_and_nullable_enums_keep_types_apart(ws):
	schema = {"type": "object", "properties": {"is_spam": {"type": "boolean"}, "level": {"type": ["integer", "null"], "enum": [1, 2, None]}}}
	examples = [
		{"id": "1", "input": "a", "output": {"is_spam": True, "level": 1}},
		{"id": "2", "input": "b", "output": {"is_spam": 1, "level": True}},
		{"id": "3", "input": "c", "output": {"is_spam": False, "level": None}},
	]
	result = run_stats(ws, project_with(ws, schema, examples))
	assert counts(result, "is_spam") == {"True": 1, "False": 1}
	assert counts(result, "level") == {"1": 1, "2": 0, "None": 1}


def test_invalid_schema_still_reports_duplicates(ws):
	examples = [{"id": "1", "input": "same", "output": 1}, {"id": "2", "input": "same", "output": 1}]
	result = run_stats(ws, project_with(ws, {"type": "nonsense"}, examples))
	assert result["fields"] == [] and result["refs"] == []
	assert result["duplicates"] == {"groups": 1, "conflicts": 0}


def test_duplicates_normalize_whitespace_and_flag_conflicts(ws):
	examples = [
		{"id": "1", "input": "Refund  my\norder", "output": {"label": "billing", "x": 1}},
		{"id": "2", "input": " Refund my order ", "output": {"x": 1, "label": "billing"}},
		{"id": "3", "input": "Login fails", "output": {"label": "technical"}},
		{"id": "4", "input": "Login fails", "output": {"label": "account"}},
		{"id": "5", "input": "login fails", "output": {"label": "account"}},
		{"id": "6", "input": "  ", "output": {"label": "other"}},
		{"id": "7", "input": "", "output": {"label": "other"}},
		{"id": "8", "input": "", "image": "a0a0a0a0a0a0a0a0a0a0a0a0.jpg", "output": {"label": "billing"}},
		{"id": "9", "input": " ", "image": "a0a0a0a0a0a0a0a0a0a0a0a0.jpg", "output": {"label": "technical"}},
		{"id": "10", "input": "close up", "image": "a0a0a0a0a0a0a0a0a0a0a0a0.jpg", "output": {"label": "billing"}},
	]
	project = project_with(ws, {"type": "object"}, examples)
	assert project.examples().duplicate_groups() == [
		{"ids": ["1", "2"], "conflict": False},
		{"ids": ["3", "4"], "conflict": True},
		{"ids": ["8", "9"], "conflict": True},
	]
	assert run_stats(ws, project)["duplicates"] == {"groups": 3, "conflicts": 2}


def test_ref_stats_top_distinct_and_missing(ws, tmp_path):
	source = tmp_path / "codes.txt"
	source.write_text("V1\nV2\n", encoding="utf-8")
	ws.library.create("Codes", source, "codes.txt")
	examples = [{"input": "a", "output": {"b": ["V1", "V2"]}}, {"input": "b", "output": {"b": ["V1", "V9"]}}, {"input": "c", "output": {"b": ["V1"]}}]
	[ref] = run_stats(ws, project_with(ws, CODES, examples))["refs"]
	assert (ref["field"], ref["dataset"], ref["name"]) == ("b", "codes", "Codes")
	assert (ref["distinct"], ref["size"], ref["total"], ref["missing"]) == (3, 2, 5, 1)
	assert ref["top"][0] == {"value": "V1", "count": 3}


def test_input_length_stats(ws):
	examples = [{"input": "x" * n, "output": {}} for n in (10, 20, 3600, 40)]
	project = project_with(ws, {"type": "object"}, examples, training={"max_seq_length": 1000})
	result = run_stats(ws, project)
	assert result["long_inputs"] == 1 and result["input_chars"]["max"] == 3600
	project.save_config({"training": {"max_seq_length": 0}})
	assert run_stats(ws, project)["long_inputs"] is None


def test_checker_marks_errors_and_reports_progress(ws):
	examples = [{"id": "ok", "input": "a", "output": {"label": "billing"}}, {"id": "bad", "input": "b", "output": {"label": "nope"}}, {"id": "empty", "input": "", "output": {"label": "other"}}]
	project = project_with(ws, template("classification"), examples)
	checker = Checker()
	checker.ensure(project, ws.library)
	checker.wait(project.id)
	assert checker.progress(project.id) == {"running": False, "done": 3, "total": 3, "error": None}
	assert project.examples().problem_count() == 2
	assert project.examples().get("empty")["errors"] == ["Input is empty"]


def test_key_change_rechecks_and_single_edit_is_checked_now(ws, monkeypatch):
	monkeypatch.setattr(checks, "BATCH", 1)
	examples = [{"id": f"e{i}", "input": f"t{i}", "output": {"label": "billing"}} for i in range(5)]
	project = project_with(ws, template("classification"), examples)
	checker = Checker()
	checker.ensure(project, ws.library)
	project.save_config({"schema": template("sentiment")})  # every example is now wrong
	key = checker.ensure(project, ws.library)
	checker.wait(project.id)
	assert key == checks_key(project.load_config(), {})
	assert project.examples().checked_count(key) == 5 and project.examples().problem_count() == 5
	project.examples().update("e1", {"output": {"sentiment": "positive", "score": 3}})
	assert checker.check_now(project, ws.library, ["e1"]) == {"e1": []}
	assert project.examples().problem_count() == 4


def test_ref_stats_scan_rows_in_order(ws):
	project = project_with(ws, CODES, [{"input": "a", "output": {"b": ["V1"]}}])
	sql, params = checks._ref_source(checks.ref_fields(CODES)[0])
	with project.examples().connect() as db:
		steps = " | ".join(r[3] for r in db.execute(f"EXPLAIN QUERY PLAN {sql}", params))
	assert "SCAN examples" in steps, steps


def test_a_new_checker_with_nothing_left_to_check_is_not_running(ws, monkeypatch):
	# After a restart a fresh checker starts a thread that finds nothing to do; until it exits,
	# "running" used to refuse new runs with "Still checking examples (3 of 3)".
	examples = [{"id": f"e{i}", "input": f"t{i}", "output": {"label": "billing"}} for i in range(3)]
	project = project_with(ws, template("classification"), examples)
	first = Checker()
	first.ensure(project, ws.library)
	first.wait(project.id)
	gate = threading.Event()
	real = ExampleStore.iter_unchecked

	def held(self, key, batch=500):
		gate.wait(5)
		yield from real(self, key, batch)

	monkeypatch.setattr(ExampleStore, "iter_unchecked", held)
	second = Checker()
	second.ensure(project, ws.library)
	try:
		assert second.progress(project.id) == {"running": False, "done": 3, "total": 3, "error": None}
	finally:
		gate.set()


def test_a_fresh_edit_is_not_overwritten_by_an_older_batch(ws, monkeypatch):
	project = project_with(ws, template("classification"), [{"id": "e0", "input": "t", "output": {"label": "nope"}}])
	judged, release = threading.Event(), threading.Event()
	real, main = checks.example_errors, threading.get_ident()

	def held(*args):
		errors = real(*args)
		if threading.get_ident() != main:  # the background batch has judged the old content
			judged.set()
			release.wait(5)
		return errors

	monkeypatch.setattr(checks, "example_errors", held)
	checker = Checker()
	checker.ensure(project, ws.library)
	assert judged.wait(5)
	project.examples().update("e0", {"output": {"label": "billing"}})
	assert checker.check_now(project, ws.library, ["e0"]) == {"e0": []}
	release.set()
	checker.wait(project.id)
	assert project.examples().get("e0")["errors"] == []
