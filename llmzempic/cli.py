"""Command line: `llmzempic` (web UI), `llmzempic train`, `llmzempic try`, `llmzempic decide`, `llmzempic serve`,
`llmzempic export`, `llmzempic doctor`."""

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
	from .checks import Checker
	from .fileio import StatusFile
	from .runs import create_run
	from .trainer import run_training

	ws = Workspace(args.workspace)
	project = ws.resolve(args.project)
	checker = Checker()
	checker.ensure(project, ws.library)
	checker.wait(project.id)
	run_dir = create_run(project, ws.library, checker)
	print(f"Run folder: {run_dir}")
	status = StatusFile(run_dir / "status.json")
	run_training(run_dir, status)
	from .fileio import read_json
	from .datasets import run_resolver
	from .metrics import label_breakdown
	from .ref_metrics import ref_scores
	evaluation = read_json(run_dir / "eval.json", None) or {}
	spec = read_json(run_dir / "run.json", {})
	items = evaluation.get("items") or []
	print(json.dumps({
		**(status.data.get("metrics") or {}),
		"labels": label_breakdown(items, spec.get("schema")),
		"refs": ref_scores(items, spec.get("schema"), run_resolver(run_dir)),
	}, indent=2))


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


def _decision_source(args):
	"""(run folder or None, trained questions) for decide/serve, checked before PyTorch loads."""
	from pathlib import Path

	from .decisions import DecisionError
	from .fileio import read_json

	if bool(args.run) == bool(args.model):
		raise DecisionError("Give either --run <run folder> (a decision run) or --model <base model> (training-free).")
	if not args.run:
		return None, []
	run_dir = Path(args.run).expanduser().resolve()
	spec = read_json(run_dir / "run.json", None)
	if spec is None:
		raise DecisionError(f"No training run in {run_dir}.")
	return run_dir, spec.get("questions") or []


def _load_questions(text):
	from pathlib import Path

	path = Path(text).expanduser()
	try:
		is_file = path.is_file()
	except OSError:  # inline JSON longer than a file name can be
		is_file = False
	value = json.loads(path.read_text(encoding="utf-8") if is_file else text)
	return value["questions"] if isinstance(value, dict) and "questions" in value else value


def cmd_decide(args):
	from .decision_engine import load_reader
	from .decisions import build_response, parse_request

	run_dir, trained = _decision_source(args)
	body = {"input": " ".join(args.input)}
	if args.questions:
		body["questions"] = _load_questions(args.questions)
	if args.min_confidence is not None:
		body["min_confidence"] = args.min_confidence
	parse_request(body, trained)  # report a bad request before loading the model
	reader, trained, temperature, name = load_reader(run_dir, args.model)
	state, questions, threshold = parse_request(body, trained)
	scores, tokens = reader.score(state, questions)
	print(json.dumps(build_response(name, questions, scores, tokens, temperature, threshold), indent=2, ensure_ascii=False))


def cmd_serve(args):
	import uvicorn

	from .decision_engine import load_reader
	from .decision_server import create_decision_app

	run_dir, _ = _decision_source(args)
	reader, trained, temperature, name = load_reader(run_dir, args.model)
	app = create_decision_app(reader, args.name or name, trained, temperature)
	host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
	print(f"Decisions API for {args.name or name} at http://{host}:{args.port}/v1/decisions", flush=True)
	uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


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


def cmd_package(args):
	from pathlib import Path

	from .packages import export_package

	ws = Workspace(args.workspace)
	project = ws.resolve(args.project)
	out = Path(args.out or f"{project.id}.llmz").expanduser()
	export_package(ws, project, out, args.include_run)
	print(f"Wrote {out}")


def cmd_import_package(args):
	from pathlib import Path

	from .packages import import_package

	project = import_package(Workspace(args.workspace), Path(args.file).expanduser())
	print(f"Imported as project '{project.id}'")


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

	decide = sub.add_parser("decide", help="Answer typed questions with probabilities (decision runs, or any base model)")
	decide.add_argument("input", nargs="+", help="The input text")
	decide.add_argument("--run", help="Decision run folder (…/projects/<id>/runs/<run>)")
	decide.add_argument("--model", help="Base model id or folder, read training-free (needs --questions)")
	decide.add_argument("--questions", help="JSON file or inline JSON: a list of questions (default: the run's questions)")
	decide.add_argument("--min-confidence", type=float, help="Withhold answers less confident than this (0-1)")
	decide.set_defaults(func=cmd_decide)

	serve = sub.add_parser("serve", help="Serve a Decisions API (POST /v1/decisions)")
	serve.add_argument("--run", help="Decision run folder (…/projects/<id>/runs/<run>)")
	serve.add_argument("--model", help="Base model id or folder, read training-free")
	serve.add_argument("--name", help="Model name in responses (default: the project name)")
	serve.add_argument("--host", default="127.0.0.1")
	serve.add_argument("--port", type=int, default=7871)
	serve.set_defaults(func=cmd_serve)

	package = sub.add_parser("package", help="Save a project (examples, images, data sets) as a .llmz file")
	package.add_argument("project", help="Project id, name, or folder")
	package.add_argument("--out", help="File to write (default: <project>.llmz)")
	package.add_argument("--include-run", action="store_true", help="Also include the latest trained run")
	package.set_defaults(func=cmd_package)

	import_ = sub.add_parser("import-package", help="Create a project from a .llmz file")
	import_.add_argument("file", help="Package file")
	import_.set_defaults(func=cmd_import_package)

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
		if type(e).__name__ in ("RunError", "KeyError", "ImageError", "FileNotFoundError", "PackageError", "DecisionError", "JSONDecodeError"):
			print(f"error: {e}", file=sys.stderr)
			return 1
		raise
	return 0


if __name__ == "__main__":
	sys.exit(main())
