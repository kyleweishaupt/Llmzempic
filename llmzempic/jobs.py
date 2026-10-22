"""Runs training/export in a child process so the UI stays responsive and GPU memory is freed after."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .fileio import StatusFile, read_json

PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


class JobBusy(Exception):
	pass


def job_paths(kind, run_dir):
	run_dir = Path(run_dir)
	if kind == "train":
		return run_dir / "status.json", run_dir / "train.log"
	if kind == "export":
		return run_dir / "export" / "status.json", run_dir / "export" / "export.log"
	if kind == "predict":
		return run_dir / "predict" / "status.json", run_dir / "predict" / "predict.log"
	raise ValueError(f"Unknown job kind: {kind}")


def kill_tree(proc):
	if proc.poll() is not None:
		return
	if os.name == "nt":
		subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
		return
	try:
		os.killpg(proc.pid, signal.SIGTERM)
		proc.wait(timeout=15)
	except ProcessLookupError:
		pass
	except subprocess.TimeoutExpired:
		os.killpg(proc.pid, signal.SIGKILL)


class Job:
	def __init__(self, kind, run_dir, proc, log):
		self.kind = kind
		self.run_dir = Path(run_dir)
		self.proc = proc
		self.log = log
		self.status_path = job_paths(kind, run_dir)[0]
		self.started = time.time()
		self.stopping = False

	def describe(self):
		return {"kind": self.kind, "run_id": self.run_dir.name, "project_id": self.run_dir.parent.parent.name, "started": self.started}


class JobManager:
	"""One job at a time: training and export both want the whole GPU."""

	def __init__(self, command=None):
		self._lock = threading.Lock()
		self._job = None
		self.command = command or [sys.executable, "-m", "llmzempic.worker"]

	def _reap(self):
		job = self._job
		if not job or job.proc.poll() is None:
			return
		status = read_json(job.status_path, {}) or {}
		if status.get("state") in ("queued", "running"):
			if job.stopping:
				StatusFile(job.status_path).update(state="stopped", stage="Stopped")
			else:
				StatusFile(job.status_path).update(
					state="failed", stage="Failed",
					error=f"Worker exited with code {job.proc.returncode}. See the log for details.",
				)
		job.log.close()
		self._job = None

	def current(self):
		with self._lock:
			self._reap()
			return self._job

	def start(self, kind, run_dir, env=None):
		with self._lock:
			self._reap()
			if self._job:
				raise JobBusy(f"A {self._job.kind} job is already running. Wait for it or stop it first.")
			status_path, log_path = job_paths(kind, run_dir)
			log_path.parent.mkdir(parents=True, exist_ok=True)
			log = open(log_path, "ab")
			child_env = os.environ.copy()
			child_env.update(env or {})
			child_env["PYTHONUNBUFFERED"] = "1"
			child_env["PYTHONIOENCODING"] = "utf-8"
			child_env["PYTHONPATH"] = os.pathsep.join(p for p in (PACKAGE_PARENT, child_env.get("PYTHONPATH")) if p)
			kwargs = {}
			if os.name == "nt":
				kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
			else:
				kwargs["start_new_session"] = True
			# Written before launch so a fast worker's own status updates are never overwritten.
			StatusFile(status_path).update(state="running", stage="Starting", started=time.time(), error=None)
			try:
				proc = subprocess.Popen(
					[*self.command, kind, str(run_dir)],
					stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
					env=child_env, **kwargs,
				)
			except OSError as e:
				log.close()
				StatusFile(status_path).update(state="failed", stage="Failed", error=f"Could not start the worker: {e}")
				raise
			self._job = Job(kind, run_dir, proc, log)
			return self._job

	def stop(self, run_dir=None):
		with self._lock:
			job = self._job
			if not job or (run_dir is not None and Path(run_dir) != job.run_dir):
				return False
			job.stopping = True
		kill_tree(job.proc)
		try:
			job.proc.wait(timeout=30)
		except subprocess.TimeoutExpired:
			job.proc.kill()
		with self._lock:
			self._reap()
		return True


def mark_interrupted(workspace):
	"""Runs left 'running' by a previous session can't be resumed; flag them as interrupted."""
	for project in workspace.projects():
		for run_id in project.run_ids():
			for kind in ("train", "export", "predict"):
				path = job_paths(kind, project.runs_dir / run_id)[0]
				status = read_json(path, None)
				if status and status.get("state") in ("queued", "running"):
					StatusFile(path).update(
						state="failed", stage="Interrupted",
						error="Interrupted: Llmzempic was closed while this job was running.",
					)
