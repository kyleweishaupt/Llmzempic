"""Try a trained run on new inputs before exporting it. Runs inside the worker process.

Loads the base model with the run's LoRA adapter (no merge), answers each input from
predict/request.json with greedy decoding and writes predict/result.json.
"""

import os

from .fileio import read_json, read_jsonl, write_json
from .metrics import parse_generated
from .prompts import FALLBACK_CHAT_TEMPLATE, format_output
from .schema_tools import validate_output
from .trainer import Plan, dtype_kwarg, generate, hf_token, log


def load_trained(spec, adapter_dir, plan):
	import transformers
	from peft import PeftModel
	from transformers import AutoModelForCausalLM, AutoTokenizer

	# The trainer saved the tokenizer with its pad token and chat template next to the adapter.
	tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
	kwargs = {"token": hf_token(), dtype_kwarg(transformers): plan.dtype}
	if plan.four_bit:
		from transformers import BitsAndBytesConfig
		kwargs["quantization_config"] = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_use_double_quant=True,
			bnb_4bit_compute_dtype=plan.dtype,
		)
	if plan.cuda:
		kwargs["device_map"] = {"": 0}
	model = AutoModelForCausalLM.from_pretrained(spec["base_model"], **kwargs)
	model = PeftModel.from_pretrained(model, str(adapter_dir))
	model.eval()
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if not getattr(tokenizer, "chat_template", None):
		tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
	return model, tokenizer


def run_prediction(run_dir, status):
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
	spec = read_json(run_dir / "run.json")
	request = read_json(run_dir / "predict" / "request.json")
	inputs = [str(x) for x in request["inputs"]]

	status.update(state="running", stage="Loading PyTorch")
	import torch
	plan = Plan(spec, torch)
	log(f"Base model: {spec['base_model']} with the run's LoRA adapter")
	status.update(stage="Loading the trained model", plan=plan.describe())
	model, tokenizer = load_trained(spec, run_dir / "adapter", plan)

	# Room for the longest answer seen in training, like the evaluation after training.
	rows = read_jsonl(run_dir / "train.jsonl") + read_jsonl(run_dir / "eval.jsonl")
	answer_tokens = max((len(tokenizer(format_output(r["output"]), add_special_tokens=False)["input_ids"]) for r in rows), default=256)
	max_new = min(4096, int(answer_tokens * 1.5) + 32)

	status.update(stage=f"Answering {len(inputs)} input(s)")
	log(f"Answering {len(inputs)} input(s), up to {max_new} new tokens each")
	texts = generate(model, tokenizer, spec["system_prompt"], inputs, max_new, 4, torch)
	items = []
	for text, generated in zip(inputs, texts):
		value, ok = parse_generated(generated)
		items.append({
			"input": text,
			"generated": generated,
			"value": value,
			"json_valid": ok,
			"errors": validate_output(value, spec["schema"]) if ok else ["Response is not valid JSON"],
		})
	write_json(run_dir / "predict" / "result.json", {"items": items})
	log("Done")
	status.update(state="completed", stage="Completed")
