"""Command line: `llmzempic` (web UI), `llmzempic train`, `llmzempic try`, `llmzempic export`, `llmzempic doctor`."""

import argparse
import json
import sys
import threading
import webbrowser

from .workspace import Workspace


def cmd_ui(args):
	import uvicorn

	from .server import create_app

	app = create_app(Workspace(args.workspace))
	url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
	print(f"Llmzempic is running at {url}  (workspace: {app.state.workspace.root})")
	if not args.no_browser:
		threading.Timer(1.0, lambda: webbrowser.open(url)).start()
	uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_train(args):
	from .fileio import StatusFile
	from .runs import create_run
	from .trainer import run_training

	project = Workspace(args.workspace).resolve(args.project)
	run_dir = create_run(project)
	print(f"Run folder: {run_dir}")
	status = StatusFile(run_dir / "status.json")
	run_training(run_dir, status)
	from .fileio import read_json
	from .metrics import label_breakdown
	evaluation = read_json(run_dir / "eval.json", None) or {}
	spec = read_json(run_dir / "run.json", {})
	print(json.dumps({**(status.data.get("metrics") or {}), "labels": label_breakdown(evaluation.get("items") or [], spec.get("schema"))}, indent=2))


def cmd_try(args):
	from pathlib import Path

	from .fileio import StatusFile, read_json
	from .runs import prepare_prediction

	run_dir = Path(args.run).expanduser().resolve()
	payload = {"inputs": args.inputs}
	if args.image:
		from .images import save_image
		names = [save_image(run_dir / "images", Path(p).expanduser().read_bytes()) for p in args.image]
		# Text is optional for image runs; pair each image with its text, or none.
		payload = {"inputs": (args.inputs + [""] * len(names))[:len(names)], "images": names}
	# Validate the run and inputs before loading PyTorch.
	predict_dir = prepare_prediction(run_dir, payload)
	from .predictor import run_prediction
	run_prediction(run_dir, StatusFile(predict_dir / "status.json"))
	print(json.dumps(read_json(predict_dir / "result.json"), indent=2, ensure_ascii=False))


def cmd_export(args):
	from pathlib import Path

	from .exporter import run_export
	from .fileio import StatusFile
	from .runs import prepare_export

	ws = Workspace(args.workspace)
	run_dir = Path(args.run).expanduser().resolve()
	export_dir = prepare_export(run_dir, {"gguf": args.gguf, "keep_merged": args.keep_merged, "ollama": args.ollama, "ollama_name": args.name}, ws.load_settings(), ws.tools_dir)
	status = StatusFile(export_dir / "status.json")
	run_export(run_dir, status)
	print(f"Exported to {export_dir}")


def cmd_doctor(args):
	from .hardware import detect
	print(json.dumps(detect(), indent=2))


def main(argv=None):
	parser = argparse.ArgumentParser(prog="llmzempic", description="Fine-tune open-source LLMs for structured output.")
	parser.add_argument("--workspace", help="Folder for projects and runs (default: ~/Llmzempic or $LLMZEMPIC_HOME)")
	sub = parser.add_subparsers(dest="command")

	ui = sub.add_parser("ui", help="Start the web UI (default)")
	ui.add_argument("--host", default="127.0.0.1")
	ui.add_argument("--port", type=int, default=7870)
	ui.add_argument("--no-browser", action="store_true")
	ui.set_defaults(func=cmd_ui)

	train = sub.add_parser("train", help="Train a project without the UI")
	train.add_argument("project", help="Project id, name, or folder")
	train.set_defaults(func=cmd_train)

	export = sub.add_parser("export", help="Export a completed run")
	export.add_argument("run", help="Run folder (…/projects/<id>/runs/<run>)")
	export.add_argument("--gguf", default="q8_0", help="q8_0 (default), f16, bf16, q4_k_m, q5_k_m or none")
	export.add_argument("--name", help="Model name (default: project name)")
	export.add_argument("--keep-merged", action="store_true", help="Also keep the merged safetensors model")
	export.add_argument("--ollama", action="store_true", help="Register the model with Ollama")
	export.set_defaults(func=cmd_export)

	try_ = sub.add_parser("try", help="Answer inputs with a trained run (before exporting it)")
	try_.add_argument("run", help="Run folder (…/projects/<id>/runs/<run>)")
	try_.add_argument("inputs", nargs="*", help="One or more input texts (optional with --image)")
	try_.add_argument("--image", action="append", metavar="PATH", help="Image file to answer (image runs; repeatable)")
	try_.set_defaults(func=cmd_try)

	doctor = sub.add_parser("doctor", help="Show detected hardware and libraries")
	doctor.set_defaults(func=cmd_doctor)

	argv = sys.argv[1:] if argv is None else list(argv)
	args = parser.parse_args(argv)
	if not args.command:
		args = parser.parse_args(argv + ["ui"])
	try:
		args.func(args)
	except KeyboardInterrupt:
		return 130
	except Exception as e:  # surface run/setup errors without a traceback wall
		if type(e).__name__ in ("RunError", "KeyError", "ImageError", "FileNotFoundError"):
			print(f"error: {e}", file=sys.stderr)
			return 1
		raise
	return 0


if __name__ == "__main__":
	sys.exit(main())
