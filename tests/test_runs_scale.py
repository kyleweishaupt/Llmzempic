import pytest

from llmzempic import checks
from llmzempic.checks import Checker
from llmzempic.datasets import run_resolver
from llmzempic.fileio import read_json, read_jsonl
from llmzempic.runs import RunError, create_run
from llmzempic.workspace import Workspace

CODES = {
	"type": "object",
	"properties": {
		"label": {"type": "string", "enum": ["a", "b"]},
		"b": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}},
	},
	"required": ["label", "b"],
}


class BusyChecker(Checker):
	def ensure(self, project, library, force=False):
		return "key"

	def progress(self, project_id):
		return {"running": True, "done": 5, "total": 20}


@pytest.fixture
def ws(tmp_path):
	ws = Workspace(tmp_path / "ws")
	source = tmp_path / "codes.txt"
	source.write_text("V1  First\nV2  Second\n", encoding="utf-8")
	ws.library.create("Codes", source, "codes.txt")
	return ws


def make_project(ws, n=20, bad=0):
	project = ws.create("Coder")
	project.save_config({"schema": CODES})
	rows = [{"id": f"e{i}", "input": f"note {i}", "output": {"label": "ab"[i % 2], "b": ["V1", "V2"][: 1 + i % 2]}} for i in range(n)]
	rows += [{"id": f"bad{i}", "input": f"bad {i}", "output": {"label": "a", "b": ["V9"]}} for i in range(bad)]
	project.examples().import_records(rows, "append")
	return project


def ready_checker(ws, project):
	checker = Checker()
	checker.ensure(project, ws.library)
	checker.wait(project.id)
	return checker


def test_run_snapshots_dataset_and_prompt_has_no_values(ws):
	project = make_project(ws)
	run_dir = create_run(project, ws.library, ready_checker(ws, project))
	spec = read_json(run_dir / "run.json")
	assert (run_dir / "datasets" / "codes" / "values.db").exists()
	assert spec["datasets"]["codes"]["name"] == "Codes" and spec["datasets"]["codes"]["rows"] == 2
	assert 'One value from the data set \\"Codes\\".' in spec["system_prompt"]
	assert "V1" not in spec["system_prompt"]


def test_replacing_dataset_leaves_run_snapshot(ws, tmp_path):
	project = make_project(ws)
	run_dir = create_run(project, ws.library, ready_checker(ws, project))
	source = tmp_path / "new.txt"
	source.write_text("V3\n", encoding="utf-8")
	ws.library.replace("codes", source, "new.txt")
	assert run_resolver(run_dir).missing("codes", ["V1", "V3"]) == {"V3"}
	assert ws.library.resolver().missing("codes", ["V1", "V3"]) == {"V1"}


def test_refuses_while_checking(ws):
	project = make_project(ws)
	with pytest.raises(RunError, match=r"Still checking examples \(5 of 20\)"):
		create_run(project, ws.library, BusyChecker())


def test_refuses_with_problems(ws):
	project = make_project(ws, bad=3)
	with pytest.raises(RunError, match="3 example"):
		create_run(project, ws.library, ready_checker(ws, project))


def test_refuses_missing_dataset(ws):
	project = make_project(ws)
	project.save_config({"schema": {**CODES, "properties": {**CODES["properties"], "c": {"type": "string", "x-dataset": "gone"}}}})
	with pytest.raises(RunError, match='Data set "gone" no longer exists'):
		create_run(project, ws.library, ready_checker(ws, project))


def test_streams_split_to_files(ws):
	project = make_project(ws, n=2000)
	run_dir = create_run(project, ws.library, ready_checker(ws, project))
	spec = read_json(run_dir / "run.json")
	train, held_out = read_jsonl(run_dir / "train.jsonl"), read_jsonl(run_dir / "eval.jsonl")
	assert (len(train), len(held_out)) == (spec["train_count"], spec["eval_count"]) == (1800, 200)
	assert {r["id"] for r in train}.isdisjoint(r["id"] for r in held_out)
	assert "errors" not in train[0] and spec["training"]["eval_max"] == 300


def test_refuses_examples_left_unchecked(ws, monkeypatch):
	# A checker thread that died leaves rows with no errors recorded; they must not count as valid.
	project = make_project(ws)
	monkeypatch.setattr(checks, "example_errors", lambda *args: 1 / 0)
	checker = Checker()
	checker.ensure(project, ws.library)
	checker.wait(project.id)
	assert "division by zero" in checker.progress(project.id)["error"]
	with pytest.raises(RunError, match="Checking examples failed: division by zero"):
		create_run(project, ws.library, checker)
	monkeypatch.undo()
	checker.ensure(project, ws.library, force=True)
	checker.wait(project.id)
	assert create_run(project, ws.library, checker).is_dir()


def test_a_finished_checker_picks_up_rows_left_unchecked(ws):
	project = make_project(ws)
	checker = ready_checker(ws, project)
	with project.examples().connect() as db:  # e.g. an edit saved but never checked
		db.execute("UPDATE examples SET checked = NULL WHERE id = 'e3'")
		db.commit()
	checker.ensure(project, ws.library)
	checker.wait(project.id)
	assert checker.progress(project.id) == {"running": False, "done": 20, "total": 20, "error": None}
