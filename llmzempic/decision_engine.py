"""Decision models on a causal LM: training items, the KL-anchored trainer, and reading the probability of
every option of every question from one batched forward pass. Runs where PyTorch is loaded (the worker,
`llmzempic decide` and `llmzempic serve`); the PyTorch imports stay inside the functions.
"""

import contextlib
import inspect
import random
import time

from . import decisions
from .fileio import read_json
from .metrics import score_predictions
from .prompts import build_messages, render_chat, tokenize_answer
from .schema_tools import compact_json, validate_output


class DecisionReader:
	"""Scores each option's answer number (all of its tokens) right after the question's prompt. Questions
	and the options sharing a token prefix are batched into one forward pass that keeps only the logits of
	the last few positions, so a request costs about one prompt's worth of compute per question."""

	def __init__(self, model, tokenizer, instructions="", batch_rows=16):
		self.model = model
		self.tokenizer = tokenizer
		self.system_prompt = decisions.system_prompt(instructions)
		self.batch_rows = batch_rows
		forward = model.get_base_model().forward if hasattr(model, "get_base_model") else model.forward
		params = inspect.signature(forward).parameters
		self.keep_arg = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)
		self.pass_positions = "position_ids" in params
		self._paths = {}

	def prompt_ids(self, state, question):
		text = render_chat(self.tokenizer, build_messages(self.system_prompt, decisions.user_prompt(state, question)), True)
		return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

	def paths(self, count):
		"""Token ids of each answer number, tokenized the way training tokenizes the answer."""
		if count not in self._paths:
			paths = [list(self.tokenizer(n, add_special_tokens=False)["input_ids"]) for n in decisions.option_ids(count)]
			if not all(paths) or len({tuple(p) for p in paths}) != count:
				raise RuntimeError("The tokenizer can't tell the answer numbers apart.")
			self._paths[count] = paths
		return self._paths[count]

	def score(self, state, questions, adapter=True):
		"""([log-probability of each option's number] per question, tokens read). adapter=False reads the
		base model with the LoRA adapter switched off (the training-free baseline)."""
		import torch

		rows, needs, index = [], [], {}
		for qi, q in enumerate(questions):
			ids = self.prompt_ids(state, q)
			for oi, path in enumerate(self.paths(len(q["options"]))):
				# Options whose numbers share all but the last token are read from the same row.
				key = (qi, tuple(path[:-1]))
				if key not in index:
					index[key] = len(rows)
					rows.append(ids + path[:-1])
				needs.append((index[key], len(ids) - 1, path, qi, oi))
		keep = max(len(n[2]) for n in needs)
		by_row = {}
		for need in needs:
			by_row.setdefault(need[0], []).append(need)
		scores = [[0.0] * len(q["options"]) for q in questions]
		pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
		device = self.model.device
		switch = self.model.disable_adapter() if not adapter else contextlib.nullcontext()
		with switch, torch.no_grad():
			for start in range(0, len(rows), self.batch_rows):
				chunk = rows[start:start + self.batch_rows]
				width = max(map(len, chunk))
				# Left padding puts every row's answer positions in the last `keep` columns.
				ids = torch.tensor([[pad] * (width - len(r)) + r for r in chunk], device=device)
				mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in chunk], device=device)
				kwargs = {"input_ids": ids, "attention_mask": mask}
				if self.pass_positions:
					kwargs["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0)
				if self.keep_arg:
					kwargs[self.keep_arg] = keep
				logp = torch.log_softmax(self.model(**kwargs).logits[:, -keep:, :].float(), dim=-1)
				at_row, at_pos, at_token, slots = [], [], [], []
				for local, row in enumerate(chunk):
					for _, pos, path, qi, oi in by_row.get(start + local, ()):
						for k, token in enumerate(path):
							at_row.append(local)
							at_pos.append(keep - len(row) + pos + k)
							at_token.append(token)
							slots.append((qi, oi))
				values = logp[at_row, at_pos, at_token].tolist()
				for (qi, oi), value in zip(slots, values):
					scores[qi][oi] += value
		return scores, sum(map(len, rows))


# Training

def training_items(rows, tokenizer, spec, on_progress):
	"""([(row index, question index, gold option)] for every question each row answers, token lengths)."""
	questions = spec["questions"]
	items, lengths = [], []
	for i, row in enumerate(rows):
		for qi, gold in decisions.gold_items(row["output"], questions):
			q = questions[qi]
			order = decisions.shuffled_order(q)
			feature = tokenize_answer(tokenizer, spec["system_prompt"], decisions.user_prompt(row["input"], q, order), decisions.answer_id(q, gold, order))
			items.append((i, qi, gold))
			lengths.append(len(feature["input_ids"]))
		if (i + 1) % 500 == 0:
			on_progress(i + 1, len(rows))
	on_progress(len(rows), len(rows))
	return items, lengths


class DecisionFeatures:
	"""One training sequence per (example, question): the prompt, then the gold option's number. With a
	seed, choice options are reshuffled every time a sequence is drawn."""

	def __init__(self, rows, items, tokenizer, spec, seed=None):
		self.rows = rows
		self.items = list(items)
		self.tokenizer = tokenizer
		self.questions = spec["questions"]
		self.system_prompt = spec["system_prompt"]
		self.rng = random.Random(seed) if seed is not None else None

	def __len__(self):
		return len(self.items)

	def __getitem__(self, k):
		i, qi, gold = self.items[k]
		q = self.questions[qi]
		order = decisions.shuffled_order(q, self.rng)
		user = decisions.user_prompt(self.rows[i]["input"], q, order)
		return tokenize_answer(self.tokenizer, self.system_prompt, user, decisions.answer_id(q, gold, order))


def anchor_positions(labels, attention_mask, per_row, torch):
	"""Up to per_row evenly spaced positions per row whose next token is part of the prompt."""
	nxt = (labels[:, 1:] == -100) & (attention_mask[:, 1:] == 1)
	rows, cols = [], []
	for r in range(nxt.shape[0]):
		idx = nxt[r].nonzero().flatten()
		if idx.numel() > per_row:
			idx = idx[torch.linspace(0, idx.numel() - 1, per_row, device=idx.device).long()]
		rows.append(torch.full_like(idx, r))
		cols.append(idx)
	return torch.cat(rows), torch.cat(cols)


def anchored_trainer(base, kl_weight, per_row=32):
	"""A Trainer whose loss adds kl_weight * KL(base model || tuned model) over next-token predictions on
	the prompt text, with the base read by switching the adapter off. The decision answer is learned while
	the model's general language predictions stay anchored to where they started."""
	import torch
	import torch.nn.functional as F

	class AnchoredTrainer(base):
		def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
			loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
			rows, cols = anchor_positions(inputs["labels"], inputs["attention_mask"], per_row, torch)
			if rows.numel():
				with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
					base_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[rows, cols].float()
				tuned = outputs.logits[rows, cols].float()
				kl = F.kl_div(F.log_softmax(tuned, -1), F.log_softmax(base_logits, -1), log_target=True, reduction="batchmean")
				loss = loss + kl_weight * kl.to(loss.dtype)
			return (loss, outputs) if return_outputs else loss

	return AnchoredTrainer


# Evaluation and answers

def run_temperature(run_dir):
	return float((read_json(run_dir / "calibration.json", {}) or {}).get("temperature") or 1.0)


def _trim(report):
	keep = ("count", "accuracy", "nll", "brier", "ece")
	return {
		**{k: report[k] for k in keep},
		"questions": [{"name": q["name"], "accuracy": q["accuracy"], "ece": q["ece"]} for q in report["questions"]],
	}


def evaluate(model, tokenizer, spec, rows, log, baseline=True):
	"""Score held-out rows: accuracy, calibration (ECE, Brier, NLL) before and after temperature scaling,
	per-question results, latency, and the base model reading the same prompts without the adapter."""
	questions = spec["questions"]
	reader = DecisionReader(model, tokenizer, spec.get("instructions", ""))
	samples, base_samples, latencies, tokens, all_scores = [], [], [], [], []
	for n, row in enumerate(rows, start=1):
		started = time.perf_counter()
		scores, read = reader.score(row["input"], questions)
		latencies.append((time.perf_counter() - started) * 1000)
		tokens.append(read)
		all_scores.append(scores)
		golds = decisions.gold_items(row["output"], questions)
		samples += [{"question": qi, "scores": scores[qi], "gold": gold} for qi, gold in golds]
		if baseline:
			base_scores = reader.score(row["input"], questions, adapter=False)[0]
			base_samples += [{"question": qi, "scores": base_scores[qi], "gold": gold} for qi, gold in golds]
		if n % 50 == 0:
			log(f"Scored {n} of {len(rows)}")
	report = decisions.calibration_report(samples, questions)
	temperature = report["temperature"]
	texts, answers = [], []
	for row, scores in zip(rows, all_scores):
		probs = [decisions.softmax(s, temperature) for s in scores]
		expected = row["output"] if isinstance(row["output"], dict) else {}
		texts.append(compact_json({q["name"]: decisions.predicted_value(q, p) for q, p in zip(questions, probs) if q["name"] in expected}))
		answers.append([decisions.build_answer(q, p) for q, p in zip(questions, probs)])
	evaluation = score_predictions(rows, texts, spec["schema"])
	for item, answer in zip(evaluation["items"], answers):
		item["decisions"] = answer
	report["latency_ms"] = {"p50": round(decisions.percentile(latencies, 0.5), 1), "p95": round(decisions.percentile(latencies, 0.95), 1)} if latencies else None
	report["input_tokens"] = round(sum(tokens) / len(tokens)) if tokens else 0
	report["baseline"] = _trim(decisions.calibration_report(base_samples, questions)) if base_samples else None
	evaluation["decisions"] = report
	return evaluation


def predict_items(model, tokenizer, spec, temperature, inputs, name):
	"""Try-it answers: the Decisions API response per input, plus the answer as a project output value."""
	questions = spec["questions"]
	reader = DecisionReader(model, tokenizer, spec.get("instructions", ""))
	items = []
	for text in inputs:
		scores, read = reader.score(text, questions)
		value = {q["name"]: decisions.predicted_value(q, decisions.softmax(s, temperature)) for q, s in zip(questions, scores)}
		items.append({
			"input": text,
			"image": None,
			"generated": compact_json(value),
			"value": value,
			"json_valid": True,
			"errors": validate_output(value, spec["schema"]),
			"decisions": decisions.build_response(name, questions, scores, read, temperature),
		})
	return items


def load_reader(run_dir=None, model_id=None):
	"""(reader, trained questions, temperature, model name) for a decision run, or for any base model
	read training-free (no questions are trained; every request brings its own)."""
	import torch

	from .runs import model_name
	from .trainer import Plan, hf_token, model_kwargs

	if run_dir is not None:
		from .predictor import load_trained

		spec = read_json(run_dir / "run.json")
		if not decisions.is_decision_spec(spec):
			raise decisions.DecisionError("This run answers with JSON. Serve a run trained with the Decisions answer, or use --model for a base model.")
		status = read_json(run_dir / "status.json", {}) or {}
		if status.get("state") != "completed" or not (run_dir / "adapter").exists():
			raise decisions.DecisionError("Only completed training runs can answer decisions.")
		model, tokenizer = load_trained(spec, run_dir / "adapter", Plan(spec, torch))
		return DecisionReader(model, tokenizer, spec.get("instructions", "")), spec["questions"], run_temperature(run_dir), model_name(spec)

	from transformers import AutoModelForCausalLM, AutoTokenizer

	from .prompts import FALLBACK_CHAT_TEMPLATE
	from .workspace import catalog_entry

	# With a known size, "auto" quantization only goes 4-bit when the model wouldn't fit in 16-bit.
	entry = catalog_entry(model_id)
	plan = Plan({"training": {}, "input_kind": "text", "params_b": entry["params_b"] if entry else None}, torch)
	tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token())
	model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs(plan))
	model.eval()
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if not getattr(tokenizer, "chat_template", None):
		tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
	return DecisionReader(model, tokenizer), [], 1.0, model_id
