"""Creating training runs (snapshots of a project) and summarizing them for the UI."""

import json
import re
import time

from . import __version__
from .dataset import split_examples
from .fileio import read_json, write_json, write_jsonl
from .images import image_path, link_images
from .prompts import build_system_prompt
from .schema_tools import check_schema, validate_output
from .workspace import catalog_entry, slugify

GGUF_TYPES = ["q8_0", "f16", "bf16", "q4_k_m", "q5_k_m", "none"]


class RunError(Exception):
	pass


def _input_errors(example, input_kind, images_dir):
	if input_kind != "image":
		return [] if str(example.get("input", "")).strip() else ["Input is empty"]
	if not example.get("image"):
		return ["No image"]
	try:
		image_path(images_dir, example["image"])
	except KeyError:
		return ["Image file is missing"]
	return []


def validate_examples(examples, schema, input_kind="text", images_dir=None):
	invalid = []
	if check_schema(schema):
		return invalid
	for example in examples:
		errors = _input_errors(example, input_kind, images_dir) + validate_output(example.get("output"), schema)
		if errors:
			invalid.append({"id": example.get("id"), "errors": errors})
	return invalid


def model_name(spec):
	"""Name used for the exported model (Ollama tag, file names)."""
	return slugify(spec.get("project_name") or spec.get("project_id") or "model")


def create_run(project):
	config = project.load_config()
	schema = config["schema"]
	problems = check_schema(schema)
	if problems:
		raise RunError(problems[0])
	if not str(config.get("base_model") or "").strip():
		raise RunError("Choose a base model first.")
	input_kind = config.get("input_kind", "text")
	entry = catalog_entry(config["base_model"].strip())
	if entry and input_kind == "image" and not entry.get("vision"):
		raise RunError(f"{entry['name']} can't read images. Pick a vision model on the Model tab.")
	if entry and input_kind != "image" and entry.get("vision"):
		raise RunError(f"{entry['name']} is a vision model. Pick a text model on the Model tab.")
	examples = project.load_examples()
	if len(examples) < 2:
		raise RunError("Add at least 2 examples (20 or more is recommended).")
	invalid = validate_examples(examples, schema, input_kind, project.images_dir)
	if invalid:
		raise RunError(f"{len(invalid)} example(s) have problems. Fix them on the Examples tab.")

	training = config["training"]
	train_rows, eval_rows, split = split_examples(
		examples, schema, float(training.get("eval_fraction") or 0), int(training.get("seed") or 42)
	)

	spec = {
		"project_id": project.id,
		"project_name": config.get("name") or project.id,
		"base_model": config["base_model"].strip(),
		"params_b": entry["params_b"] if entry else None,
		"input_kind": input_kind,
		"instructions": config.get("instructions", ""),
		"schema": schema,
		"system_prompt": build_system_prompt(
			config.get("instructions", ""), schema, config.get("include_schema_in_prompt", True)
		),
		"training": training,
		"train_count": len(train_rows),
		"eval_count": len(eval_rows),
		"split": split,
		"created": time.time(),
		"app_version": __version__,
	}
	run_dir = project.new_run_dir()
	if input_kind == "image":
		link_images([e["image"] for e in examples], project.images_dir, run_dir / "images")
	write_json(run_dir / "run.json", spec)
	write_jsonl(run_dir / "train.jsonl", train_rows)
	write_jsonl(run_dir / "eval.jsonl", eval_rows)
	write_json(run_dir / "status.json", {"state": "queued", "stage": "Waiting to start", "updated": time.time()})
	return run_dir


def prepare_export(run_dir, options, settings, tools_dir):
	status = read_json(run_dir / "status.json", {}) or {}
	if status.get("state") != "completed" or not (run_dir / "adapter").exists():
		raise RunError("Only completed training runs can be exported.")
	gguf = str(options.get("gguf") or "q8_0").lower()
	if gguf not in GGUF_TYPES:
		raise RunError(f"Unknown GGUF type '{gguf}'.")
	spec = read_json(run_dir / "run.json", {})
	name = str(options.get("ollama_name") or model_name(spec)).strip().lower()
	if not re.match(r"^[a-z0-9][a-z0-9._:/-]*$", name):
		raise RunError("Model names may only use lowercase letters, digits and . _ : / -")
	export_dir = run_dir / "export"
	export_dir.mkdir(exist_ok=True)
	write_json(export_dir / "options.json", {
		"gguf": gguf,
		"keep_merged": bool(options.get("keep_merged")),
		"ollama": bool(options.get("ollama")),
		"ollama_name": name,
		"llama_cpp_dir": settings.get("llama_cpp_dir") or "",
		"tools_dir": str(tools_dir),
	})
	write_json(export_dir / "status.json", {"state": "queued", "stage": "Waiting to start", "updated": time.time()})
	return export_dir


MAX_PREDICT_INPUTS = 10
MAX_PREDICT_CHARS = 200_000


def prepare_prediction(run_dir, payload, source_images=None):
	"""Validate a Try-it request. Image names refer to source_images (a project's images folder),
	or, without it, to images already in the run's images folder."""
	status = read_json(run_dir / "status.json", {}) or {}
	if status.get("state") != "completed" or not (run_dir / "adapter").exists():
		raise RunError("Only completed training runs can be tried.")
	vision = (read_json(run_dir / "run.json", {}) or {}).get("input_kind") == "image"
	inputs = payload.get("inputs")
	if not isinstance(inputs, list) or not inputs:
		raise RunError("Enter an input to try.")
	if len(inputs) > MAX_PREDICT_INPUTS:
		raise RunError(f"Try at most {MAX_PREDICT_INPUTS} inputs at a time.")
	inputs = [str(x or "") for x in inputs]
	images = payload.get("images") if vision else None
	if vision:
		if not isinstance(images, list) or len(images) != len(inputs) or not all(isinstance(x, str) and x for x in images):
			raise RunError("Add an image to try.")
		try:
			for name in images:
				image_path(source_images or run_dir / "images", name)
		except KeyError:
			raise RunError("That image is no longer available. Add it again.")
	elif any(not x.strip() for x in inputs):
		raise RunError("Enter an input to try.")
	if any(len(x) > MAX_PREDICT_CHARS for x in inputs):
		raise RunError("That input is too long to try here.")
	if images and source_images:
		link_images(images, source_images, run_dir / "images")
	predict_dir = run_dir / "predict"
	predict_dir.mkdir(exist_ok=True)
	for stale in ("result.json", "predict.log"):
		(predict_dir / stale).unlink(missing_ok=True)
	write_json(predict_dir / "request.json", {"inputs": inputs, "images": images} if vision else {"inputs": inputs})
	write_json(predict_dir / "status.json", {"state": "queued", "stage": "Waiting to start", "updated": time.time()})
	return predict_dir


def _summary_metrics(evaluation):
	if not evaluation:
		return None
	return {k: v for k, v in evaluation.items() if k != "items"}


def run_summary(run_dir):
	spec = read_json(run_dir / "run.json", {}) or {}
	status = read_json(run_dir / "status.json", {}) or {}
	export_status = read_json(run_dir / "export" / "status.json", None)
	return {
		"id": run_dir.name,
		"created": spec.get("created"),
		"base_model": spec.get("base_model"),
		"train_count": spec.get("train_count"),
		"eval_count": spec.get("eval_count"),
		"split": spec.get("split"),
		"input_kind": spec.get("input_kind", "text"),
		"model_name": model_name(spec),
		"status": status,
		"metrics": _summary_metrics(read_json(run_dir / "eval.json", None)),
		"export": export_status,
	}


def list_export_files(run_dir):
	export_dir = run_dir / "export"
	if not export_dir.exists():
		return []
	files = []
	for path in sorted(export_dir.rglob("*")):
		if path.is_file() and path.name not in ("status.json", "options.json", "export.log") and not path.name.endswith(".tmp"):
			files.append({"path": str(path.relative_to(export_dir)).replace("\\", "/"), "bytes": path.stat().st_size})
	return files


def read_events(run_dir, since=0):
	path = run_dir / "events.jsonl"
	if not path.exists():
		return []
	rows = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			try:
				rows.append(json.loads(line))
			except ValueError:
				break  # the worker may be mid-write on the last line
	return rows[since:]
