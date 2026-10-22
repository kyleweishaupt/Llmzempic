import sys

from llmzempic.cli import main
from llmzempic.fileio import write_json


def test_try_rejects_bad_runs_before_loading_the_model(tmp_path, capsys):
	run_dir = tmp_path / "run"
	run_dir.mkdir()
	write_json(run_dir / "status.json", {"state": "queued"})
	assert main(["try", str(run_dir), "hello"]) == 1
	assert "completed" in capsys.readouterr().err

	write_json(run_dir / "status.json", {"state": "completed"})
	(run_dir / "adapter").mkdir()
	assert main(["try", str(run_dir), "   "]) == 1
	assert "Enter an input" in capsys.readouterr().err
	assert "llmzempic.predictor" not in sys.modules
