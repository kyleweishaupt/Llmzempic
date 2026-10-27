"""Command line: `llmzempic` (web UI), `llmzempic train`, `llmzempic try`, `llmzempic decide`, `llmzempic eval`,
`llmzempic serve`, `llmzempic mcp`, `llmzempic export`, `llmzempic doctor`."""

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
	"""(run or export folder or None, trained questions) for decide, checked before PyTorch loads."""
	from pathlib import Path

	from .decision_engine import decision_source
	from .decisions import DecisionError

	if bool(args.run) == bool(args.model):
		raise DecisionError("Give either --run <run or export folder> (a decision model) or --model <base model> (training-free).")
	if not args.run:
		return None, []
	path = Path(args.run).expanduser().resolve()
	return path, decision_source(path)[0]["questions"]


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
	from .decision_engine import load_model
	from .decisions import DecisionError, parse_request

	run_dir, trained = _decision_source(args)
	body = {"input": " ".join(args.input)}
	if args.image:
		import base64
		from pathlib import Path

		image = base64.b64encode(Path(args.image).expanduser().read_bytes()).decode()
		body["input"] = [{"role": "user", "content": body["input"], "images": [image]}]
	if args.questions:
		body["questions"] = _load_questions(args.questions)
	if args.min_confidence is not None:
		body["min_confidence"] = args.min_confidence if args.min_confidence == "auto" else float(args.min_confidence)
	if args.samples != 1:
		body["samples"] = args.samples
	parse_request(body, trained)  # report a bad request before loading the model
	served = load_model(run_dir, args.model)
	state, questions, settings = parse_request(body, served.questions)
	if served.vision and settings["image"] is None:
		raise DecisionError("This model decides about images: add --image.")
	print(json.dumps(served.answer(state, questions, settings), indent=2, ensure_ascii=False))


def _eval_rows(args, questions):
	"""Labeled rows {"id", "input", "output"} from --data (JSONL or a JSON array) or a Hugging Face split."""
	from pathlib import Path

	from .decisions import DecisionError

	if bool(args.data) == bool(args.hub):
		raise DecisionError("Give either --data <file> or --hub <dataset>.")
	if args.data:
		text = Path(args.data).expanduser().read_text(encoding="utf-8")
		records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
		rows = [{"id": str(r.get("id") or i), "input": r["input"], "output": r["output"]} for i, r in enumerate(records)]
		return rows[:args.limit] if args.limit else rows
	from .hub_import import import_examples

	mapping = dict(item.partition("=")[::2] for item in args.label) if args.label else {}
	mapping = {q: (c or q) for q, c in mapping.items()} or {q["name"]: q["name"] for q in questions}
	examples, _, skipped, _ = import_examples(args.hub, args.config or "default", args.split, args.text, list(mapping.values()), args.limit or 5000)
	back = {c: q for q, c in mapping.items()}
	if skipped:
		print(f"Skipped {skipped} row(s) without a text or a label.", file=sys.stderr)
	return [{"id": str(i), "input": e["input"], "output": {back[c]: v for c, v in e["output"].items()}} for i, e in enumerate(examples)]


def cmd_eval(args):
	"""Score a decision run or export on a labeled set: accuracy, calibration, the base model, and how many
	inputs nearly repeat a training example."""
	from pathlib import Path

	from .decision_engine import decision_source, evaluate, load_model
	from .decisions import DecisionError
	from .fileio import JsonlRows, write_json
	from .overlap import find_overlaps

	path = Path(args.run).expanduser().resolve()
	spec, _ = decision_source(path)
	rows = _eval_rows(args, spec["questions"])
	if not rows:
		raise DecisionError("No labeled rows to score.")
	overlap = find_overlaps(rows, JsonlRows(path / "train.jsonl")) if (path / "train.jsonl").exists() else None
	if overlap and overlap["count"]:
		print(f"{overlap['count']} of {len(rows)} input(s) nearly repeat a training example" + (" (left out)." if args.drop_overlap else "."), file=sys.stderr)
		if args.drop_overlap:
			leaked = set(overlap["held_out_ids"])
			rows = [r for r in rows if r["id"] not in leaked]
	served = load_model(path)
	evaluation = evaluate(served.reader, spec, rows, lambda message: print(message, file=sys.stderr), baseline=args.baseline, temperature=served.temperature)
	report = evaluation["decisions"]
	summary = {
		"model": served.name, "count": len(rows), "answers": report["count"],
		"accuracy": report["accuracy"], "all_right": evaluation["exact_match"],
		"served": report["at_temperature"], "ece_uncalibrated": report["ece"],
		"questions": [{k: q[k] for k in ("name", "type", "count", "accuracy", "ece", "auto") if k in q} for q in report["questions"]],
		"baseline": report["baseline"], "latency_ms": report["latency_ms"], "input_tokens": report["input_tokens"],
		"output_tokens_avoided": report["output_tokens_avoided"],
		"overlap": overlap["count"] if overlap else None,
	}
	if args.out:
		write_json(Path(args.out).expanduser(), {**evaluation, "summary": summary, "overlap": overlap})
	print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_mcp(args):
	"""A decision model as an MCP tool over stdio; stdout carries only protocol messages."""
	import contextlib

	from .decision_engine import load_model
	from .decision_mcp import run_stdio

	run_dir, _ = _decision_source(args)
	stdout = sys.stdout
	with contextlib.redirect_stdout(sys.stderr):
		served = load_model(run_dir, args.model, args.name)
	print(f"MCP server ready: {served.name}", file=sys.stderr, flush=True)
	run_stdio(served, sys.stdin, stdout)


def cmd_serve(args):
	from pathlib import Path

	import uvicorn

	from .decision_engine import decision_source, load_model
	from .decision_monitor import SERVE_DIR, DecisionLog
	from .decision_server import Ensemble, Route, create_decision_app, parse_named, parse_route, unique_names
	from .decisions import DecisionError

	sources = [(name, Path(value).expanduser().resolve(), None) for name, value in parse_named(args.run)]
	sources += [(name, None, value) for name, value in parse_named(args.model)]
	if not sources:
		raise DecisionError("Give at least one --run <run or export folder> or --model <base model>.")
	ensembles = [parse_route(text, ensemble=True) for text in args.ensemble]
	routes = [parse_route(text) for text in args.ab]
	for _, path, _ in sources:
		if path is not None:
			decision_source(path)  # report a bad folder before loading anything
	shared = {}
	models = unique_names([load_model(path, model_id, name, shared) for name, path, model_id in sources])
	if args.name and len(models) == 1:
		models[0].name = args.name
	by_name = {m.name: m for m in models}

	def member(name):
		if name not in by_name:
			raise DecisionError(f'No served model is called "{name}". Served: {", ".join(by_name)}.')
		return by_name[name]

	for name, members in ensembles:
		by_name[name] = Ensemble(name, [member(n) for n, _ in members])
	served = list(by_name.values())
	ab = [Route(name, [member(n) for n, _ in members], [w for _, w in members]) for name, members in routes]
	logs = {}
	if not args.no_log:
		for m in served:
			folder = Path(args.log_dir).expanduser() / m.name if args.log_dir else getattr(m, "source", None) and m.source / SERVE_DIR
			if folder:
				logs[m.name] = DecisionLog(folder, log_inputs=not args.no_log_inputs)
	app = create_decision_app(served, ab, logs, args.default)
	host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
	print(f"Decisions API at http://{host}:{args.port}/v1/decisions serving " + ", ".join([*by_name, *(r.name for r in ab)]), flush=True)
	for name, log in logs.items():
		print(f"  {name}: decisions logged in {log.folder}", flush=True)
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
	decide.add_argument("input", nargs="*", help="The input text (optional with --image)")
	decide.add_argument("--image", metavar="PATH", help="Image to decide about (image models)")
	decide.add_argument("--run", help="Decision run or export folder")
	decide.add_argument("--model", help="Base model id or folder, read training-free (needs --questions)")
	decide.add_argument("--questions", help="JSON file or inline JSON: a list of questions (default: the run's questions)")
	decide.add_argument("--min-confidence", help='Withhold answers less confident than this (0-1), or "auto" for 90%% held-out accuracy')
	decide.add_argument("--samples", type=int, default=1, help="Read choice questions in this many option orders and average (1-8)")
	decide.set_defaults(func=cmd_decide)

	evaluate = sub.add_parser("eval", help="Score a decision run or export on a labeled file or Hugging Face split")
	evaluate.add_argument("run", help="Decision run or export folder")
	evaluate.add_argument("--data", help="JSONL (or JSON array) of {\"input\", \"output\"} rows")
	evaluate.add_argument("--hub", metavar="DATASET", help="Hugging Face dataset id, e.g. mteb/banking77")
	evaluate.add_argument("--config", help="Dataset config (default: default)")
	evaluate.add_argument("--split", default="test", help="Dataset split (default: test)")
	evaluate.add_argument("--text", default="text", help="Text column (default: text)")
	evaluate.add_argument("--label", action="append", metavar="QUESTION=COLUMN", help="Label column per question (default: columns named like the questions)")
	evaluate.add_argument("--limit", type=int, default=0, help="Score at most this many rows")
	evaluate.add_argument("--baseline", action="store_true", help="Also score the base model without the adapter")
	evaluate.add_argument("--drop-overlap", action="store_true", help="Leave out inputs that nearly repeat a training example")
	evaluate.add_argument("--out", help="Write every prediction and the summary to this JSON file")
	evaluate.set_defaults(func=cmd_eval)

	mcp = sub.add_parser("mcp", help="Run a decision model as an MCP tool server over stdio (for agents)")
	mcp.add_argument("--run", help="Decision run or export folder")
	mcp.add_argument("--model", help="Base model id or folder, read training-free")
	mcp.add_argument("--name", help="Model name (default: the project name)")
	mcp.set_defaults(func=cmd_mcp)

	serve = sub.add_parser("serve", help="Serve a Decisions API (POST /v1/decisions)")
	serve.add_argument("--run", action="append", default=[], metavar="[NAME=]FOLDER", help="Decision run or export folder (repeatable)")
	serve.add_argument("--model", action="append", default=[], metavar="[NAME=]MODEL", help="Base model id or folder, read training-free (repeatable)")
	serve.add_argument("--name", help="Model name in responses when serving one model (default: the project name)")
	serve.add_argument("--ab", action="append", default=[], metavar="ROUTE=A:90,B:10", help="A/B route: split requests between served models by weight")
	serve.add_argument("--ensemble", action="append", default=[], metavar="NAME=A,B", help="Serve the average of several models' probabilities")
	serve.add_argument("--default", help="Model or route for requests that don't name one (default: the first route, else the first model)")
	serve.add_argument("--log-dir", help="Log decisions in <dir>/<model> (default: each run or export folder's serve/ folder)")
	serve.add_argument("--no-log", action="store_true", help="Don't log decisions")
	serve.add_argument("--no-log-inputs", action="store_true", help="Log decisions without their input text")
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
		if type(e).__name__ in ("RunError", "KeyError", "ImageError", "FileNotFoundError", "PackageError", "DecisionError", "JSONDecodeError", "HubError"):
			print(f"error: {e}", file=sys.stderr)
			return 1
		raise
	return 0


if __name__ == "__main__":
	sys.exit(main())
