"""LoRA / QLoRA supervised fine-tuning that teaches a model to answer with schema-valid JSON.

Runs inside the worker process. Loss is computed only on the assistant's JSON answer, the
system prompt (instructions + schema) is identical to the one shipped with the export, and
the held-out examples are scored for JSON validity, schema validity and exact match.
"""

import importlib.util
import inspect
import math
import os
import shutil
import time

from . import decision_engine, decision_monitor, decisions, vision
from .fileio import JsonlRows, append_jsonl, read_json, read_jsonl, write_json
from .hardware import estimate_memory_gb
from .metrics import score_predictions
from .overlap import find_overlaps
from .prompts import FALLBACK_CHAT_TEMPLATE, build_messages, format_output, render_chat, tokenize_example
from .train_data import LazyFeatures, measure_lengths, select_within

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(message):
	print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def hf_token():
	return os.environ.get("HF_TOKEN") or None


def filter_kwargs(fn, kwargs):
	params = inspect.signature(fn).parameters
	return {k: v for k, v in kwargs.items() if k in params}


def dtype_kwarg(transformers):
	"""transformers 4.56 renamed from_pretrained(torch_dtype=) to dtype=."""
	from packaging.version import Version
	return "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


class Plan:
	def __init__(self, spec, torch):
		t = spec["training"]
		self.cuda = torch.cuda.is_available()
		mps = getattr(torch.backends, "mps", None)
		self.device = "cuda" if self.cuda else ("mps" if mps and mps.is_available() else "cpu")
		has_bnb = importlib.util.find_spec("bitsandbytes") is not None
		quant = t.get("quantization", "auto")
		self.four_bit = False
		if self.cuda and has_bnb:
			if quant == "4bit":
				self.four_bit = True
			elif quant == "auto":
				vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
				params = spec.get("params_b")
				self.four_bit = params is None or estimate_memory_gb(params, False) > vram * 0.85
		elif quant == "4bit":
			log("4-bit needs an NVIDIA GPU with bitsandbytes installed; training in 16-bit instead.")

		accel = t.get("accelerator", "auto")
		self.vision = vision.is_vision(spec)
		# Decision training reads logits (KL anchor, option scores), which Unsloth's fused loss doesn't return.
		self.decisions = decisions.is_decision_spec(spec)
		self.unsloth = (
			self.cuda and not self.vision and not self.decisions and accel in ("auto", "unsloth")
			and importlib.util.find_spec("unsloth") is not None
		)
		if accel == "unsloth" and not self.unsloth:
			reason = (
				"isn't used for image projects" if self.vision else "isn't used for decision models" if self.decisions
				else "needs an NVIDIA GPU and `pip install unsloth`"
			)
			log(f"Unsloth was requested but {reason}; using standard PEFT.")

		if self.cuda and torch.cuda.is_bf16_supported():
			self.precision, self.dtype = "bf16", torch.bfloat16
		elif self.cuda:
			self.precision, self.dtype = "fp16", torch.float16
		else:
			self.precision, self.dtype = "fp32", torch.float32

		if self.four_bit:
			self.optim = "paged_adamw_8bit"
		elif self.cuda:
			self.optim = "adamw_torch_fused"
		else:
			self.optim = "adamw_torch"

	def describe(self):
		return {
			"device": self.device,
			"precision": self.precision,
			"four_bit": self.four_bit,
			"unsloth": bool(self.unsloth),
			"optimizer": self.optim,
		}


class PadCollator:
	"""Dynamic padding to the longest sequence in each batch (no wasted compute on fixed-length padding)."""

	def __init__(self, pad_id, multiple=None):
		self.pad_id = pad_id
		self.multiple = multiple

	def __call__(self, features):
		import torch
		longest = max(len(f["input_ids"]) for f in features)
		if self.multiple:
			longest = math.ceil(longest / self.multiple) * self.multiple
		ids, mask, labels = [], [], []
		for f in features:
			pad = longest - len(f["input_ids"])
			ids.append(f["input_ids"] + [self.pad_id] * pad)
			mask.append(f["attention_mask"] + [0] * pad)
			labels.append(f["labels"] + [-100] * pad)
		return {
			"input_ids": torch.tensor(ids, dtype=torch.long),
			"attention_mask": torch.tensor(mask, dtype=torch.long),
			"labels": torch.tensor(labels, dtype=torch.long),
		}


def model_kwargs(plan):
	"""from_pretrained() arguments for the training/inference device, dtype and 4-bit plan."""
	import transformers

	kwargs = {"token": hf_token(), dtype_kwarg(transformers): plan.dtype}
	if plan.four_bit:
		from transformers import BitsAndBytesConfig
		kwargs["quantization_config"] = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_use_double_quant=True,
			bnb_4bit_compute_dtype=plan.dtype,
			# The vision encoder is small and frozen; quantizing it only costs image understanding.
			llm_int8_skip_modules=["lm_head", "vision", "visual", "connector", "projector", "merger"] if plan.vision else None,
		)
	if plan.cuda:
		kwargs["device_map"] = {"": 0}
	return kwargs


def load_standard(spec, plan):
	from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
	from transformers import AutoModelForCausalLM, AutoTokenizer

	t = spec["training"]
	if plan.vision:
		processor = vision.load_processor(spec["base_model"], hf_token())
		model = vision.load_model(spec["base_model"], model_kwargs(plan))
	else:
		processor = AutoTokenizer.from_pretrained(spec["base_model"], token=hf_token())
		model = AutoModelForCausalLM.from_pretrained(spec["base_model"], **model_kwargs(plan))
	if plan.four_bit:
		model = prepare_model_for_kbit_training(
			model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
		)
	lora = LoraConfig(
		r=int(t["lora_r"]),
		lora_alpha=int(t["lora_alpha"]),
		lora_dropout=float(t["lora_dropout"]),
		bias="none",
		task_type="CAUSAL_LM",
		target_modules="all-linear",
		exclude_modules=vision.VISION_MODULES if plan.vision else None,
	)
	return get_peft_model(model, lora), processor


def load_unsloth(spec, plan):
	from unsloth import FastLanguageModel

	t = spec["training"]
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=spec["base_model"],
		max_seq_length=int(t.get("max_seq_length") or 4096),
		dtype=None,
		load_in_4bit=plan.four_bit,
		token=hf_token(),
	)
	model = FastLanguageModel.get_peft_model(
		model,
		r=int(t["lora_r"]),
		lora_alpha=int(t["lora_alpha"]),
		lora_dropout=float(t["lora_dropout"]),
		target_modules=LORA_TARGETS,
		bias="none",
		use_gradient_checkpointing="unsloth",
		random_state=int(t.get("seed") or 42),
	)
	return model, tokenizer


def generate(model, tokenizer, system_prompt, inputs, max_new_tokens, batch_size, torch):
	tokenizer.padding_side = "left"
	gen_config = getattr(model, "generation_config", None)
	if gen_config is not None:
		gen_config.do_sample = False
		for attr in ("temperature", "top_p", "top_k"):
			setattr(gen_config, attr, None)
	texts = []
	for i in range(0, len(inputs), batch_size):
		prompts = [render_chat(tokenizer, build_messages(system_prompt, x), True) for x in inputs[i:i + batch_size]]
		enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
		with torch.no_grad():
			out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
		texts += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
	return texts


def answer(model, processor, spec, rows, images_dir, max_new_tokens, batch_size, torch):
	"""Greedy answers for rows of {"input"} (text runs) or {"input", "image"} (image runs)."""
	if vision.is_vision(spec):
		return vision.generate(model, processor, spec["system_prompt"], rows, images_dir, max_new_tokens, batch_size, torch)
	return generate(model, processor, spec["system_prompt"], [r["input"] for r in rows], max_new_tokens, batch_size, torch)


def feature_length(feature):
	return feature["length"] if "length" in feature else len(feature["input_ids"])


def run_training(run_dir, status):
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
	spec = read_json(run_dir / "run.json")
	t = spec["training"]
	train_rows = JsonlRows(run_dir / "train.jsonl")  # can be hundreds of thousands of long rows
	eval_rows = read_jsonl(run_dir / "eval.jsonl")  # capped at a few thousand by the split
	system_prompt = spec["system_prompt"]
	events_path = run_dir / "events.jsonl"

	if eval_rows:
		status.update(state="running", stage="Checking held-out examples for near-copies")
		overlap = find_overlaps(eval_rows, train_rows)
		write_json(run_dir / "overlap.json", overlap)
		if overlap["count"]:
			log(f"{overlap['count']} held-out example(s) nearly repeat a training example; scores on them may be optimistic (overlap.json).")

	status.update(state="running", stage="Loading PyTorch")
	import torch
	plan = Plan(spec, torch)
	if plan.unsloth:
		import unsloth  # noqa: F401  (must be imported before transformers to apply its kernels)
	import transformers
	from transformers import Trainer, TrainerCallback, TrainingArguments

	transformers.set_seed(int(t.get("seed") or 42))
	log(f"Base model: {spec['base_model']}")
	log(f"Plan: {plan.describe()}")
	status.update(stage="Loading base model (first use downloads it)", plan=plan.describe())
	model, processor = load_unsloth(spec, plan) if plan.unsloth else load_standard(spec, plan)
	tokenizer = processor.tokenizer if plan.vision else processor
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if not plan.vision and not getattr(tokenizer, "chat_template", None):
		log("Base model has no chat template; using a simple '### Role:' template.")
		tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
	tokenizer.padding_side = "right"

	max_len = int(t.get("max_seq_length") or 0)
	if plan.decisions:
		# One sequence per (example, question); options are reshuffled every time a sequence is drawn.
		progress = lambda done, total: status.update(stage=f"Measuring examples ({done:,} of {total:,})")
		images_dir = run_dir / "images"
		if plan.vision:
			status.update(stage="Preparing images")
			items, measured = decision_engine.vision_training_items(train_rows, processor, spec, images_dir, progress, log)
			eval_items, eval_measured = decision_engine.vision_training_items(eval_rows, processor, spec, images_dir, lambda done, total: None, log)
		else:
			items, measured = decision_engine.training_items(train_rows, tokenizer, spec, progress)
			eval_items, eval_measured = decision_engine.training_items(eval_rows, tokenizer, spec, lambda done, total: None)
		keep, skipped = select_within(measured, max_len)
		eval_keep, eval_skipped = select_within(eval_measured, max_len)
		if skipped or eval_skipped:
			ids = sorted({train_rows[items[i][0]]["id"] for i in skipped} | {eval_rows[eval_items[i][0]]["id"] for i in eval_skipped})
			write_json(run_dir / "skipped.json", {"max_length": max_len, "ids": ids})
			log(f"Skipped {len(ids)} example(s) with a question longer than {max_len} tokens; their ids are in skipped.json.")
		# Runs from before the listwise loss existed have no decision_loss and resume with the token loss.
		listwise = t.get("decision_loss", "token") == "listwise"
		seed = int(t.get("seed") or 42)
		train_items, held_items = [items[i] for i in keep], [eval_items[i] for i in eval_keep]
		if plan.vision:
			train_features = decision_engine.VisionDecisionFeatures(train_rows, train_items, processor, spec, images_dir, seed, listwise)
			eval_features = list(decision_engine.VisionDecisionFeatures(eval_rows, held_items, processor, spec, images_dir, listwise=listwise))
			base_collator = vision.VisionCollator(processor, model)
		else:
			train_features = decision_engine.DecisionFeatures(train_rows, train_items, tokenizer, spec, seed, listwise)
			eval_features = list(decision_engine.DecisionFeatures(eval_rows, held_items, tokenizer, spec, listwise=listwise))
			base_collator = PadCollator(tokenizer.pad_token_id, 8 if plan.cuda else None)
		collator = decision_engine.DecisionCollator(base_collator)
		lengths = [measured[i] for i in keep] + [eval_measured[i] for i in eval_keep]
		answer_tokens = 8
		log(f"{len(spec['questions'])} question(s): " + ", ".join(f"{q['name']} ({q['type']}, {len(q['options'])} options)" for q in spec["questions"]))
	elif plan.vision:
		status.update(stage="Preparing images")
		images_dir = run_dir / "images"
		train_features = vision.build_features(processor, system_prompt, list(train_rows), images_dir, log)
		eval_features = vision.build_features(processor, system_prompt, eval_rows, images_dir, log)
		collator = vision.VisionCollator(processor, model)
		if max_len:
			before = len(train_features)
			train_features = [f for f in train_features if feature_length(f) <= max_len]
			eval_features = [f for f in eval_features if feature_length(f) <= max_len]
			if len(train_features) < before:
				log(f"Skipped {before - len(train_features)} training example(s) longer than {max_len} tokens.")
		lengths = [feature_length(f) for f in train_features + eval_features]
		answer_tokens = max(len(tokenizer(format_output(r["output"]), add_special_tokens=False)["input_ids"]) for r in list(train_rows) + eval_rows)
	else:
		measured = measure_lengths(tokenizer, system_prompt, train_rows, 500,
			lambda done, total: status.update(stage=f"Measuring examples ({done:,} of {total:,})"))
		keep, skipped = select_within([n for n, _ in measured], max_len)
		eval_measured = measure_lengths(tokenizer, system_prompt, eval_rows, 500, lambda done, total: None)
		eval_keep, eval_skipped = select_within([n for n, _ in eval_measured], max_len)
		if skipped or eval_skipped:
			ids = [train_rows[i]["id"] for i in skipped] + [eval_rows[i]["id"] for i in eval_skipped]
			write_json(run_dir / "skipped.json", {"max_length": max_len, "ids": ids})
			log(f"Skipped {len(ids)} example(s) longer than {max_len} tokens; their ids are in skipped.json.")
		train_features = LazyFeatures(train_rows, keep, tokenizer, system_prompt)
		eval_features = [tokenize_example(tokenizer, system_prompt, eval_rows[i]["input"], eval_rows[i]["output"]) for i in eval_keep]
		collator = PadCollator(tokenizer.pad_token_id, 8 if plan.cuda else None)
		lengths = [measured[i][0] for i in keep] + [eval_measured[i][0] for i in eval_keep]
		answer_tokens = max(a for _, a in measured + eval_measured)
	if not len(train_features):
		raise RuntimeError("Every training example is longer than the max sequence length.")
	longest = max(lengths)
	log(f"{len(train_features)} training / {len(eval_features)} eval examples, longest {longest} tokens")
	if longest > 4096:
		log("Long examples use a lot of memory; lower the batch size if you run out of memory.")
	trainable, total = 0, 0
	for p in model.parameters():
		total += p.numel()
		if p.requires_grad:
			trainable += p.numel()
	log(f"Trainable LoRA parameters: {trainable:,} of {total:,} ({100 * trainable / max(total, 1):.2f}%)")

	batch_size = int(t["batch_size"])
	grad_accum = int(t["grad_accum"])
	epochs = float(t["epochs"])
	steps_per_epoch = math.ceil(len(train_features) / (batch_size * grad_accum))
	total_steps = max(1, math.ceil(steps_per_epoch * epochs))
	args = TrainingArguments(**filter_kwargs(TrainingArguments.__init__, {
		"output_dir": str(run_dir / "checkpoints"),
		"per_device_train_batch_size": batch_size,
		"per_device_eval_batch_size": batch_size,
		"gradient_accumulation_steps": grad_accum,
		"num_train_epochs": epochs,
		"learning_rate": float(t["learning_rate"]),
		"lr_scheduler_type": "cosine",
		"warmup_steps": int(total_steps * 0.05) if total_steps >= 20 else 0,
		"weight_decay": 0.0,
		"max_grad_norm": 1.0,
		"logging_steps": 1,
		"logging_first_step": True,
		# Checkpoints let a long run resume after a crash or a stop; only the newest is kept.
		"save_strategy": "steps",
		"save_steps": max(50, total_steps // 10),
		"save_total_limit": 1,
		"eval_strategy": "epoch" if eval_features else "no",
		"bf16": plan.precision == "bf16",
		"fp16": plan.precision == "fp16",
		"gradient_checkpointing": plan.cuda and not plan.unsloth,
		"gradient_checkpointing_kwargs": {"use_reentrant": False},
		"optim": plan.optim,
		"report_to": "none",
		"seed": int(t.get("seed") or 42),
		"dataloader_num_workers": 0,
		"dataloader_pin_memory": plan.cuda,
		"remove_unused_columns": False,
		"disable_tqdm": True,
		"use_cpu": plan.device == "cpu",
	}))

	started = time.time()

	class Progress(TrainerCallback):
		def on_log(self, args, state, control, logs=None, **kwargs):
			logs = logs or {}
			elapsed = time.time() - started
			step, total = state.global_step, state.max_steps or total_steps
			event = {"step": step, "total": total, "epoch": round(state.epoch or 0, 3), "elapsed": round(elapsed, 1)}
			for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
				if key in logs:
					event[key] = logs[key]
			if "loss" not in event and "eval_loss" not in event:
				return
			append_jsonl(events_path, event)
			eta = elapsed / step * (total - step) if step else None
			status.update(stage="Training", progress={"step": step, "total": total, "eta": eta})
			if "loss" in event:
				log(f"step {step}/{total}  loss {event['loss']:.4f}")
			if "eval_loss" in event:
				log(f"epoch {event['epoch']}  eval loss {event['eval_loss']:.4f}")

	kl_weight = float(t.get("kl_weight") or 0) if plan.decisions else 0.0
	if kl_weight > 0:
		log(f"KL anchor to the base model, weight {kl_weight}")
	if plan.decisions and listwise:
		log("Listwise loss over each question's options")
	trainer = (decision_engine.decision_trainer(Trainer, kl_weight, listwise) if plan.decisions else Trainer)(
		model=model,
		args=args,
		train_dataset=train_features,
		eval_dataset=eval_features or None,
		data_collator=collator,
		callbacks=[Progress()],
	)
	try:
		from transformers.trainer_callback import PrinterCallback
		trainer.remove_callback(PrinterCallback)  # Progress already logs a cleaner line per step
	except ImportError:
		pass
	status.update(stage="Training", progress={"step": 0, "total": total_steps, "eta": None})
	resume = any((run_dir / "checkpoints").glob("checkpoint-*"))
	if resume:
		log("Resuming from the last checkpoint")
	trainer.train(resume_from_checkpoint=True if resume else None)
	train_seconds = time.time() - started

	status.update(stage="Saving adapter")
	adapter_dir = run_dir / "adapter"
	model.save_pretrained(str(adapter_dir))
	# Image runs save the whole processor: tokenizer, chat template and the capped image size.
	processor.save_pretrained(str(adapter_dir))
	shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
	log(f"Saved LoRA adapter to {adapter_dir}")

	status.update(stage="Evaluating", progress=None)
	if plan.unsloth:
		from unsloth import FastLanguageModel
		FastLanguageModel.for_inference(model)
	model.eval()
	rows, eval_on = eval_rows, "held-out examples"
	if not rows:
		rows, eval_on = [train_rows[i] for i in range(min(20, len(train_rows)))], "training examples (add 10+ examples for a held-out test)"
	max_new = min(4096, int(answer_tokens * 1.5) + 32)
	log(f"Evaluating on {len(rows)} {eval_on}")
	if plan.decisions:
		reader = decision_engine.make_reader(model, processor, spec)
		evaluation = decision_engine.evaluate(reader, spec, rows, log, run_dir / "images")
		report = evaluation["decisions"]
		if t.get("compare_generation"):
			report["generation_baseline"] = decision_engine.generation_baseline(model, processor, spec, rows, run_dir / "images", torch, log)
		labels = decision_monitor.label_counts(train_rows, spec["questions"])
		write_json(run_dir / "calibration.json", decision_engine.calibration_file(report, labels))
		base = f" (base model {report['baseline']['accuracy']:.0%})" if report["baseline"] else ""
		log(
			f"Accuracy {report['accuracy']:.0%}{base}  ECE {report['ece']:.3f}, "
			f"{report['after_calibration']['ece']:.3f} at temperature {report['temperature']}  "
			f"all questions right {evaluation['exact_match']:.0%}"
		)
	else:
		texts = answer(model, processor, spec, rows, run_dir / "images", max_new, max(1, batch_size), torch)
		evaluation = score_predictions(rows, texts, spec["schema"])
		log(f"JSON valid {evaluation['json_valid']:.0%}  schema valid {evaluation['schema_valid']:.0%}  exact match {evaluation['exact_match']:.0%}")
	evaluation["eval_on"] = eval_on
	write_json(run_dir / "eval.json", evaluation)
	status.update(
		state="completed",
		stage="Completed",
		train_seconds=round(train_seconds, 1),
		metrics={k: v for k, v in evaluation.items() if k != "items"},
	)
