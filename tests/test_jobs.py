import sys
import textwrap
import time

from llmzempic.fileio import read_json
from llmzempic.jobs import JobBusy, JobManager, job_paths

import pytest


def fake_worker(tmp_path, body):
	script = tmp_path / "fake_worker.py"
	script.write_text(textwrap.dedent(body), encoding="utf-8")
	return [sys.executable, str(script)]


def wait_for(predicate, timeout=20):
	end = time.time() + timeout
	while time.time() < end:
		if predicate():
			return True
		time.sleep(0.05)
	return False


def test_crashed_worker_is_marked_failed(tmp_path):
	jobs = JobManager(fake_worker(tmp_path, "import sys; print('boom'); sys.exit(3)"))
	run_dir = tmp_path / "run"
	run_dir.mkdir()
	jobs.start("train", run_dir)
	assert wait_for(lambda: jobs.current() is None)
	status = read_json(run_dir / "status.json")
	assert status["state"] == "failed"
	assert "code 3" in status["error"]
	assert "boom" in (run_dir / "train.log").read_text()


def test_stop_kills_worker_and_blocks_second_job(tmp_path):
	jobs = JobManager(fake_worker(tmp_path, "import time; time.sleep(60)"))
	run_dir = tmp_path / "run"
	(run_dir / "export").mkdir(parents=True)
	job = jobs.start("export", run_dir)
	assert job.describe()["kind"] == "export"
	with pytest.raises(JobBusy):
		jobs.start("train", run_dir)
	assert jobs.stop(run_dir) is True
	assert jobs.current() is None
	assert read_json(run_dir / "export" / "status.json")["state"] == "stopped"


def test_worker_status_is_kept(tmp_path):
	body = """
	import json, sys, pathlib
	path = pathlib.Path(sys.argv[2]) / "status.json"
	path.write_text(json.dumps({"state": "completed", "stage": "Completed"}))
	"""
	jobs = JobManager(fake_worker(tmp_path, body))
	run_dir = tmp_path / "run"
	run_dir.mkdir()
	jobs.start("train", run_dir)
	assert wait_for(lambda: jobs.current() is None)
	assert read_json(run_dir / "status.json")["state"] == "completed"


def test_job_paths_per_kind(tmp_path):
	assert job_paths("train", tmp_path) == (tmp_path / "status.json", tmp_path / "train.log")
	assert job_paths("export", tmp_path) == (tmp_path / "export" / "status.json", tmp_path / "export" / "export.log")
	assert job_paths("predict", tmp_path) == (tmp_path / "predict" / "status.json", tmp_path / "predict" / "predict.log")
	with pytest.raises(ValueError):
		job_paths("nope", tmp_path)

