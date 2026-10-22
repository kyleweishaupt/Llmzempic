"""Try a trained run on new inputs before exporting it. Runs inside the worker process.

Loads the base model with the run's LoRA adapter (no merge), answers each input (and image,
for image runs) from predict/request.json with greedy decoding and writes predict/result.json.
"""

import os

from . import vision
from .fileio import JsonlRows, read_json, read_jsonl, write_json
from .metrics import parse_generated
from .prompts import FALLBACK_CHAT_TEMPLATE, format_output
from .schema_tools import validate_output
from .trainer import Plan, answer, log, model_kwargs


def load_trained(spec, adapter_dir, plan):
	from peft import PeftModel
	from transformers import AutoModelForCausalLM, AutoTokenizer

	# The trainer saved the tokenizer (or processor) with its pad token and chat template next to the adapter.
	if plan.vision:
		processor = vision.load_processor(adapter_dir)
		model = vision.load_model(spec["base_model"], model_kwargs(plan))
		tokenizer = processor.tokenizer
	else:
		processor = tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
		model = AutoModelForCausalLM.from_pretrained(spec["base_model"], **model_kwargs(plan))
	model = PeftModel.from_pretrained(model, str(adapter_dir))
	model.eval()
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if not plan.vision and not getattr(tokenizer, "chat_template", None):
		tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
	return model, processor


def run_prediction(run_dir, status):
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
	spec = read_json(run_dir / "run.json")
	request = read_json(run_dir / "predict" / "request.json")
	inputs = [str(x) for x in request["inputs"]]
	images = request.get("images") or [None] * len(inputs)
	rows = [{"input": text, "image": image} for text, image in zip(inputs, images)]

	status.update(state="running", stage="Loading PyTorch")
	import torch
	plan = Plan(spec, torch)
	log(f"Base model: {spec['base_model']} with the run's LoRA adapter")
	status.update(stage="Loading the trained model", plan=plan.describe())
	model, processor = load_trained(spec, run_dir / "adapter", plan)
	tokenizer = processor.tokenizer if plan.vision else processor

	# Room for the longest answer seen in training, like the evaluation after training.
	# A sample is enough to size the answer; a run can have hundreds of thousands of training rows.
	seen = read_jsonl(run_dir / "eval.jsonl") + [row for _, row in zip(range(2000), JsonlRows(run_dir / "train.jsonl"))]
	answer_tokens = max((len(tokenizer(format_output(r["output"]), add_special_tokens=False)["input_ids"]) for r in seen), default=256)
	max_new = min(4096, int(answer_tokens * 1.5) + 32)

	status.update(stage=f"Answering {len(rows)} input(s)")
	log(f"Answering {len(rows)} input(s), up to {max_new} new tokens each")
	texts = answer(model, processor, spec, rows, run_dir / "images", max_new, 4, torch)
	items = []
	for row, generated in zip(rows, texts):
		value, ok = parse_generated(generated)
		items.append({
			"input": row["input"],
			"image": row["image"],
			"generated": generated,
			"value": value,
			"json_valid": ok,
			"errors": validate_output(value, spec["schema"]) if ok else ["Response is not valid JSON"],
		})
	write_json(run_dir / "predict" / "result.json", {"items": items})
	log("Done")
	status.update(state="completed", stage="Completed")
