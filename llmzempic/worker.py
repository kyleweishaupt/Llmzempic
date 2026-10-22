"""Child-process entry point: python -m llmzempic.worker {train|export} <run dir>."""

import sys
import traceback
from pathlib import Path

from .fileio import StatusFile
from .jobs import job_paths


def run(kind, run_dir):
	run_dir = Path(run_dir)
	status = StatusFile(job_paths(kind, run_dir)[0])
	try:
		if kind == "train":
			from .trainer import run_training
			run_training(run_dir, status)
		elif kind == "export":
			from .exporter import run_export
			run_export(run_dir, status)
		else:
			raise ValueError(f"Unknown job kind: {kind}")
	except KeyboardInterrupt:
		status.update(state="stopped", stage="Stopped")
		return 1
	except Exception as e:
		traceback.print_exc()
		status.update(state="failed", stage="Failed", error=f"{type(e).__name__}: {e}")
		return 1
	return 0


def main(argv=None):
	argv = sys.argv[1:] if argv is None else argv
	if len(argv) != 2:
		print("usage: python -m llmzempic.worker {train|export} <run dir>", file=sys.stderr)
		return 2
	return run(argv[0], argv[1])


if __name__ == "__main__":
	sys.exit(main())
