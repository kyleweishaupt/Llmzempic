import json

from llmzempic.fileio import JsonlRows, write_json
from llmzempic.prompts import tokenize_example
from llmzempic.runs import run_summary
from llmzempic.train_data import LazyFeatures, measure_lengths, select_within
from test_prompts import FakeTokenizer

ROWS = [{"id": f"r{i}", "input": "é" * i + f" row {i}", "output": {"n": i}} for i in range(5)]


def write_rows(path, rows):
	path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
	return path


def test_jsonl_rows_random_access(tmp_path):
	rows = JsonlRows(write_rows(tmp_path / "train.jsonl", ROWS))
	assert len(rows) == 5
	assert rows[3] == ROWS[3] and rows[0] == ROWS[0]
	assert [r["id"] for r in rows] == ["r0", "r1", "r2", "r3", "r4"]


def test_select_within_reports_skipped():
	assert select_within([5, 20, 7, 30], 10) == ([0, 2], [1, 3])
	assert select_within([5, 20], 0) == ([0, 1], [])


def test_lazy_features_mask_prompt_tokens(tmp_path):
	rows = JsonlRows(write_rows(tmp_path / "train.jsonl", ROWS))
	features = LazyFeatures(rows, [4, 1], FakeTokenizer(), "sys")
	assert len(features) == 2
	expected = tokenize_example(FakeTokenizer(), "sys", ROWS[1]["input"], ROWS[1]["output"])
	assert features[1] == expected
	answer = [t for t in features[1]["labels"] if t != -100]
	assert answer and features[1]["labels"][0] == -100


def test_measure_lengths_reports_total_and_answer(tmp_path):
	rows = JsonlRows(write_rows(tmp_path / "train.jsonl", ROWS))
	seen = []
	lengths = measure_lengths(FakeTokenizer(), "sys", rows, 2, lambda done, total: seen.append((done, total)))
	feature = tokenize_example(FakeTokenizer(), "sys", ROWS[2]["input"], ROWS[2]["output"])
	assert lengths[2] == (len(feature["input_ids"]), sum(1 for t in feature["labels"] if t != -100))
	assert seen[-1] == (5, 5)


def test_run_summary_resumable_when_checkpoint_exists(tmp_path):
	run_dir = tmp_path / "run"
	(run_dir / "checkpoints" / "checkpoint-10").mkdir(parents=True)
	write_json(run_dir / "status.json", {"state": "failed"})
	write_json(run_dir / "skipped.json", {"max_length": 512, "ids": ["a", "b"]})
	summary = run_summary(run_dir)
	assert summary["resumable"] is True and summary["skipped"] == 2
	write_json(run_dir / "status.json", {"state": "completed"})
	assert run_summary(run_dir)["resumable"] is False
	assert run_summary(tmp_path / "missing")["skipped"] is None
