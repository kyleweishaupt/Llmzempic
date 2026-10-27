"""Decision models on a causal LM: training items, the KL-anchored trainer, and reading the probability of
every option of every question from one batched forward pass. Runs where PyTorch is loaded (the worker,
`llmzempic decide` and `llmzempic serve`); the PyTorch imports stay inside the functions.
"""

import contextlib
import copy
import inspect
import random
import time

from . import decisions, vision
from .fileio import read_json
from .metrics import score_predictions
from .prompts import build_messages, render_chat, tokenize_answer
from .schema_tools import compact_json, validate_output


def text_completion(tokenizer, system_prompt):
	"""The answer's token ids after the prompt, exactly as training tokenizes them (text models)."""
	def completion(answer):
		return [t for t in tokenize_answer(tokenizer, system_prompt, "x", answer)["labels"] if t != -100]
	return completion


def vision_completion(processor, system_prompt):
	"""The answer's token ids after the prompt, as the chat template renders them (image models)."""
	def completion(answer):
		prompt = vision.render(processor, vision.build_messages(system_prompt, "x"), True)
		full = vision.render(processor, vision.build_messages(system_prompt, "x", answer), False)
		text = full[len(prompt):] if full.startswith(prompt) else answer + (processor.tokenizer.eos_token or "")
		return list(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
	return completion


def answer_paths(completion, count):
	"""Token ids of each answer number in the assistant's turn: the chat template may put a space or a line
	break before it (part of the path), and the end-of-turn tokens shared by every answer are left off."""
	completions = [completion(n) for n in decisions.option_ids(count)]
	shared = 0
	while all(len(c) > shared + 1 for c in completions) and len({c[len(c) - shared - 1] for c in completions}) == 1:
		shared += 1
	paths = [c[:len(c) - shared] for c in completions]
	if not all(paths) or len({tuple(p) for p in paths}) != count:
		raise RuntimeError("The tokenizer can't tell the answer numbers apart.")
	return paths


class DecisionReader:
	"""Scores each option's answer number (all of its tokens) right after the question's prompt, in one
	batched forward pass that keeps only the logits of the last few positions. The prompts of a request's
	questions all start with the system prompt and the input; that shared prefix is read once and its
	key/value cache reused for every question (text models), so the input costs one pass however many
	questions are asked."""

	MIN_PREFIX = 16

	def __init__(self, model, tokenizer, instructions="", batch_rows=16, cache=True):
		self.model = model
		self.tokenizer = tokenizer
		self.system_prompt = decisions.system_prompt(instructions)
		self.batch_rows = batch_rows
		forward = model.get_base_model().forward if hasattr(model, "get_base_model") else model.forward
		params = inspect.signature(forward).parameters
		self.keep_arg = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)
		self.pass_positions = "position_ids" in params
		self.cache = bool(cache and self.pass_positions and self.keep_arg and "past_key_values" in params)
		self._paths = {}
		self.completion = text_completion(tokenizer, self.system_prompt)

	def prompt_ids(self, state, question, order=None):
		text = render_chat(self.tokenizer, build_messages(self.system_prompt, decisions.user_prompt(state, question, order)), True)
		return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

	def paths(self, count):
		if count not in self._paths:
			self._paths[count] = answer_paths(self.completion, count)
		return self._paths[count]

	vision = False

	def score(self, state, questions, adapter=True, samples=1, image=None):
		"""([log-probability of each option] per question, usage). samples > 1 also reads choice questions
		with their options in other orders and averages the probabilities. adapter=False reads the base
		model with the LoRA adapter switched off (the training-free baseline)."""
		import torch

		if image is not None:
			raise decisions.DecisionError("This decision model reads text only; images in the input aren't supported.")

		asks = [(qi, order) for qi, q in enumerate(questions) for order in decisions.sample_orders(q, samples)]
		rows, needs, index = [], [], {}
		for ai, (qi, order) in enumerate(asks):
			ids = self.prompt_ids(state, questions[qi], order)
			for pos, path in enumerate(self.paths(len(order))):
				# Options whose numbers share all but the last token are read from the same row.
				key = (ai, tuple(path[:-1]))
				if key not in index:
					index[key] = len(rows)
					rows.append(ids + path[:-1])
				needs.append((index[key], len(ids) - 1, path, ai, order[pos]))
		switch = self.model.disable_adapter() if not adapter else contextlib.nullcontext()
		with switch, torch.no_grad():
			prefix = self.shared_prefix(rows, needs) if self.cache else 0
			try:
				reads = self.read(rows, needs, asks, questions, prefix, torch)
			except Exception:
				if not prefix:
					raise
				self.cache = False  # this model can't continue from a cache; read whole prompts from now on
				prefix = 0
				reads = self.read(rows, needs, asks, questions, 0, torch)
		per_question = [[] for _ in questions]
		for (qi, _), read in zip(asks, reads):
			per_question[qi].append(read)
		scores = [decisions.combine_samples(sets) for sets in per_question]
		return scores, {"input_tokens": sum(map(len, rows)), "cached_tokens": prefix * (len(rows) - 1)}

	def shared_prefix(self, rows, needs):
		"""Tokens every row starts with, stopping before the first position whose logits are read."""
		if len(rows) < 2:
			return 0
		limit = min(n[1] for n in needs)
		first = rows[0]
		length = 0
		while length < limit and all(len(r) > length and r[length] == first[length] for r in rows):
			length += 1
		return length if length >= self.MIN_PREFIX else 0

	def read(self, rows, needs, asks, questions, prefix, torch):
		keep = max(len(n[2]) for n in needs)
		by_row = {}
		for need in needs:
			by_row.setdefault(need[0], []).append(need)
		reads = [[0.0] * len(questions[qi]["options"]) for qi, _ in asks]
		pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
		device = self.model.device
		cached = None
		if prefix:
			head = torch.tensor([rows[0][:prefix]], device=device)
			cached = self.model(input_ids=head, attention_mask=torch.ones_like(head), use_cache=True, **{self.keep_arg: 1}).past_key_values
		for start in range(0, len(rows), self.batch_rows):
			full = rows[start:start + self.batch_rows]
			chunk = [r[prefix:] for r in full]
			width = max(map(len, chunk))
			# Left padding puts every row's answer positions in the last `keep` columns; after a cached prefix
			# the padding sits between the prefix and the rest, masked out.
			ids = torch.tensor([[pad] * (width - len(r)) + r for r in chunk], device=device)
			mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in chunk], device=device)
			kwargs = {"input_ids": ids, "attention_mask": mask}
			if cached is not None:
				cache = copy.deepcopy(cached)
				cache.batch_repeat_interleave(len(chunk))
				kwargs["past_key_values"] = cache
				kwargs["attention_mask"] = torch.cat([torch.ones(len(chunk), prefix, dtype=mask.dtype, device=device), mask], dim=1)
			if self.pass_positions:
				kwargs["position_ids"] = prefix + (mask.cumsum(-1) - 1).clamp(min=0)
			if self.keep_arg:
				kwargs[self.keep_arg] = keep
			logp = torch.log_softmax(self.model(**kwargs).logits[:, -keep:, :].float(), dim=-1)
			at_row, at_pos, at_token, slots = [], [], [], []
			for local, row in enumerate(full):
				for _, pos, path, ai, oi in by_row.get(start + local, ()):
					for k, token in enumerate(path):
						at_row.append(local)
						at_pos.append(keep - len(row) + pos + k)
						at_token.append(token)
						slots.append((ai, oi))
			values = logp[at_row, at_pos, at_token].tolist()
			for (ai, oi), value in zip(slots, values):
				reads[ai][oi] += value
		return reads


class VisionDecisionReader(DecisionReader):
	"""Reads decisions about an image (and optional text) with a vision-language model. Each question's
	prompt is processed with the image; rows of equal length share a forward pass, so nothing is padded and
	every model family computes its own positions (Qwen-VL's are 3D). No prefix cache."""

	vision = True

	def __init__(self, model, processor, instructions="", batch_rows=8):
		super().__init__(model, processor.tokenizer, instructions, batch_rows, cache=False)
		self.processor = processor
		self.completion = vision_completion(processor, self.system_prompt)
		forward = model.get_base_model().forward if hasattr(model, "get_base_model") else model.forward
		self.keys = set(inspect.signature(forward).parameters)

	def prompt_text(self, state, question, order=None):
		messages = vision.build_messages(self.system_prompt, decisions.user_prompt(state, question, order, image=True))
		return vision.render(self.processor, messages, True)

	def score(self, state, questions, adapter=True, samples=1, image=None):
		import torch

		if image is None:
			raise decisions.DecisionError("Add an image to decide on.")
		picture = vision.load_image(image)
		asks = [(qi, order) for qi, q in enumerate(questions) for order in decisions.sample_orders(q, samples)]
		rows, needs = [], []
		for ai, (qi, order) in enumerate(asks):
			encoded = self.processor(text=[self.prompt_text(state or "", questions[qi], order)], images=[[picture]], return_tensors="pt")
			length = encoded["input_ids"].shape[1]
			index = {}
			for pos, path in enumerate(self.paths(len(order))):
				key = tuple(path[:-1])
				if key not in index:
					index[key] = len(rows)
					rows.append(extend_row(encoded, list(key), torch))
				needs.append((index[key], length - 1, path, ai, order[pos]))
		reads = [[0.0] * len(questions[qi]["options"]) for qi, _ in asks]
		by_length = {}
		for i, row in enumerate(rows):
			by_length.setdefault(row["input_ids"].shape[1], []).append(i)
		by_row = {}
		for need in needs:
			by_row.setdefault(need[0], []).append(need)
		keep = max(len(n[2]) for n in needs)
		switch = self.model.disable_adapter() if not adapter else contextlib.nullcontext()
		with switch, torch.no_grad():
			for width, members in by_length.items():
				for start in range(0, len(members), self.batch_rows):
					chunk = members[start:start + self.batch_rows]
					batch = {k: torch.cat([rows[i][k] for i in chunk]).to(self.model.device) for k in rows[chunk[0]] if k in self.keys}
					if self.keep_arg:
						batch[self.keep_arg] = keep
					logp = torch.log_softmax(self.model(**batch).logits[:, -keep:, :].float(), dim=-1)
					for local, i in enumerate(chunk):
						for _, pos, path, ai, oi in by_row.get(i, ()):
							for k, token in enumerate(path):
								reads[ai][oi] += logp[local, keep - width + pos + k, token].item()
		per_question = [[] for _ in questions]
		for (qi, _), read in zip(asks, reads):
			per_question[qi].append(read)
		scores = [decisions.combine_samples(sets) for sets in per_question]
		return scores, {"input_tokens": sum(r["input_ids"].shape[1] for r in rows), "cached_tokens": 0}


def extend_row(encoded, tokens, torch):
	"""A processed one-row prompt with tokens appended: to input_ids, as 1s to attention_mask, and as 0s to
	every other per-token tensor (token type ids); image tensors are left as they are."""
	length = encoded["input_ids"].shape[1]
	row = {}
	for key, value in encoded.items():
		if not torch.is_tensor(value):
			continue
		if value.dim() == 2 and value.shape == (1, length) and tokens:
			fill = tokens if key == "input_ids" else [1] * len(tokens) if key == "attention_mask" else [0] * len(tokens)
			value = torch.cat([value, torch.tensor([fill], dtype=value.dtype)], dim=1)
		row[key] = value
	return row


def make_reader(model, processor, spec):
	if vision.is_vision(spec):
		return VisionDecisionReader(model, processor, spec.get("instructions", ""))
	return DecisionReader(model, processor, spec.get("instructions", ""))


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


def choice_steps(feature, paths, gold_pos):
	"""For the listwise loss: [logit position, gold index, candidate tokens] for each token of the gold
	answer number where more than one option is still possible (the option numbers form a token tree)."""
	gold = paths[gold_pos]
	start = next((i for i, t in enumerate(feature["labels"]) if t != -100), None)
	if start is None or feature["input_ids"][start:start + len(gold)] != gold:
		return []  # the chat template tokenized the answer differently; the token loss still applies
	steps = []
	for k, token in enumerate(gold):
		candidates = sorted({p[k] for p in paths if len(p) > k and p[:k] == gold[:k]})
		if len(candidates) > 1:
			steps.append([start + k - 1, candidates.index(token), candidates])
	return steps


class DecisionFeatures:
	"""One training sequence per (example, question): the prompt, then the gold option's number. With a
	seed, choice options are reshuffled (and sometimes cut to a subset) every time a sequence is drawn."""

	def __init__(self, rows, items, tokenizer, spec, seed=None, listwise=False):
		self.rows = rows
		self.items = list(items)
		self.tokenizer = tokenizer
		self.questions = spec["questions"]
		self.system_prompt = spec["system_prompt"]
		self.rng = random.Random(seed) if seed is not None else None
		self.listwise = listwise
		self._paths = {}
		self.completion = text_completion(tokenizer, self.system_prompt)

	def paths(self, count):
		if count not in self._paths:
			self._paths[count] = answer_paths(self.completion, count)
		return self._paths[count]

	def __len__(self):
		return len(self.items)

	def __getitem__(self, k):
		i, qi, gold = self.items[k]
		q = self.questions[qi]
		order = decisions.shuffled_order(q, self.rng, gold, decisions.SUBSET_SHARE)
		user = decisions.user_prompt(self.rows[i]["input"], q, order)
		feature = tokenize_answer(self.tokenizer, self.system_prompt, user, decisions.answer_id(q, gold, order))
		if self.listwise:
			feature["choices"] = choice_steps(feature, self.paths(len(order)), order.index(gold))
		return feature


class VisionDecisionFeatures(DecisionFeatures):
	"""DecisionFeatures for image examples: chat text with the image, processed per draw (the prompt length
	counts the image's tokens); vision.VisionCollator does the rest."""

	def __init__(self, rows, items, processor, spec, images_dir, seed=None, listwise=False):
		super().__init__(rows, items, processor.tokenizer, spec, seed, listwise)
		self.processor = processor
		self.images_dir = images_dir
		self.completion = vision_completion(processor, self.system_prompt)

	def __getitem__(self, k):
		feature, _ = self.build(k)
		return feature

	def build(self, k, shuffle=True):
		i, qi, gold = self.items[k]
		q = self.questions[qi]
		row = self.rows[i]
		order = decisions.shuffled_order(q, self.rng if shuffle else None, gold, decisions.SUBSET_SHARE)
		user = decisions.user_prompt(row.get("input") or "", q, order, image=True)
		feature, ids, aligned = vision.example_feature(
			self.processor, self.system_prompt, user, decisions.answer_id(q, gold, order), self.images_dir / row["image"])
		if self.listwise:
			labels = [-100] * feature["prompt_len"] + ids[feature["prompt_len"]:]
			feature["choices"] = choice_steps({"input_ids": ids, "labels": labels}, self.paths(len(order)), order.index(gold))
		return feature, aligned


def vision_training_items(rows, processor, spec, images_dir, on_progress, log):
	"""training_items for image examples: lengths include each image's tokens."""
	questions = spec["questions"]
	items = [(i, qi, gold) for i, row in enumerate(rows) for qi, gold in decisions.gold_items(row["output"], questions)]
	features = VisionDecisionFeatures(rows, items, processor, spec, images_dir)
	lengths, warned = [], False
	for k in range(len(items)):
		feature, aligned = features.build(k, shuffle=False)
		lengths.append(feature["length"])
		if not aligned and not warned:
			log(vision.MISALIGNED)
			warned = True
		if (k + 1) % 100 == 0:
			on_progress(k + 1, len(items))
	return items, lengths


class DecisionCollator:
	"""Pads like the base collator and adds the listwise loss's candidate tokens as flat tensors."""

	def __init__(self, base):
		self.base = base

	def __call__(self, features):
		import torch

		batch = self.base(features)
		steps = [(row, *step) for row, f in enumerate(features) for step in f.get("choices") or ()]
		if steps:
			width = max(len(s[3]) for s in steps)
			batch["choice_rows"] = torch.tensor([s[0] for s in steps], dtype=torch.long)
			batch["choice_pos"] = torch.tensor([s[1] for s in steps], dtype=torch.long)
			batch["choice_gold"] = torch.tensor([s[2] for s in steps], dtype=torch.long)
			batch["choice_tokens"] = torch.tensor([s[3] + [0] * (width - len(s[3])) for s in steps], dtype=torch.long)
			batch["choice_mask"] = torch.tensor([[1] * len(s[3]) + [0] * (width - len(s[3])) for s in steps], dtype=torch.bool)
		return batch


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


def listwise_loss(logits, steps, F):
	"""Cross-entropy over the options at each step of the option-number token tree (tree-factorized). With
	up to 9 options each number is one token and this is exactly the softmax the readout reports."""
	picked = logits[steps["choice_rows"], steps["choice_pos"]].float()
	candidates = picked.gather(1, steps["choice_tokens"]).masked_fill(~steps["choice_mask"], float("-inf"))
	return F.cross_entropy(candidates, steps["choice_gold"])


def decision_trainer(base, kl_weight=0.0, listwise=False, per_row=32):
	"""A Trainer whose loss adds, to the token loss on the answer number:
	- listwise: a cross-entropy over the question's options (listwise_loss);
	- kl_weight * KL(base model || tuned model) over next-token predictions on the prompt text, with the base
	  read by switching the adapter off. The decision is learned while the model's general language
	  predictions stay anchored to where they started."""
	import torch
	import torch.nn.functional as F

	class DecisionTrainer(base):
		def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
			steps = {k: inputs.pop(k) for k in list(inputs) if k.startswith("choice_")}
			loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
			if listwise and steps:
				loss = loss + listwise_loss(outputs.logits, steps, F).to(loss.dtype)
			if kl_weight > 0:
				rows, cols = anchor_positions(inputs["labels"], inputs["attention_mask"], per_row, torch)
				if rows.numel():
					# Every model input (images too), so the base sees exactly what the tuned model saw.
					plain = {k: v for k, v in inputs.items() if k != "labels"}
					with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
						base_logits = model(**plain).logits[rows, cols].float()
					tuned = outputs.logits[rows, cols].float()
					kl = F.kl_div(F.log_softmax(tuned, -1), F.log_softmax(base_logits, -1), log_target=True, reduction="batchmean")
					loss = loss + kl_weight * kl.to(loss.dtype)
			return (loss, outputs) if return_outputs else loss

	return DecisionTrainer


# Evaluation and answers

def run_calibration(path):
	"""(temperature, precision curves per question) saved by training, for a run or an export folder."""
	data = read_json(path / "calibration.json", {}) or {}
	return float(data.get("temperature") or 1.0), data.get("curves") or {}


def _trim(report):
	keep = ("count", "accuracy", "nll", "brier", "ece")
	return {
		**{k: report[k] for k in keep},
		"questions": [{"name": q["name"], "accuracy": q["accuracy"], "ece": q["ece"]} for q in report["questions"]],
	}


def row_image(row, images_dir):
	return images_dir / row["image"] if row.get("image") and images_dir is not None else None


def evaluate(reader, spec, rows, log, images_dir=None, baseline=True, temperature=None):
	"""Score held-out rows: accuracy, calibration (ECE, Brier, NLL) before and after temperature scaling,
	per-question results and refusal thresholds, latency, the output tokens a generated JSON answer would
	have cost, and the base model reading the same prompts without the adapter. With temperature (a served
	run's), also the metrics at that temperature."""
	questions = spec["questions"]
	schema = spec.get("schema") or decisions.questions_schema(questions)
	samples, base_samples, latencies, tokens, cached, all_scores = [], [], [], [], [], []
	for n, row in enumerate(rows, start=1):
		image = row_image(row, images_dir)
		started = time.perf_counter()
		scores, usage = reader.score(row["input"], questions, image=image)
		latencies.append((time.perf_counter() - started) * 1000)
		tokens.append(usage["input_tokens"])
		cached.append(usage["cached_tokens"])
		all_scores.append(scores)
		golds = decisions.gold_items(row["output"], questions)
		samples += [{"question": qi, "scores": scores[qi], "gold": gold} for qi, gold in golds]
		if baseline:
			base_scores = reader.score(row["input"], questions, adapter=False, image=image)[0]
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
	evaluation = score_predictions(rows, texts, schema)
	for item, answer in zip(evaluation["items"], answers):
		item["decisions"] = answer
	if temperature is not None:
		report["at_temperature"] = decisions.summarize_at(samples, temperature)
	answer_lengths = [len(reader.tokenizer(compact_json(r["output"]), add_special_tokens=False)["input_ids"]) for r in rows]
	report["output_tokens_avoided"] = round(sum(answer_lengths) / len(answer_lengths), 1) if answer_lengths else 0
	report["latency_ms"] = {"p50": round(decisions.percentile(latencies, 0.5), 1), "p95": round(decisions.percentile(latencies, 0.95), 1)} if latencies else None
	report["input_tokens"] = round(sum(tokens) / len(tokens)) if tokens else 0
	report["cached_tokens"] = round(sum(cached) / len(cached)) if cached else 0
	report["baseline"] = _trim(decisions.calibration_report(base_samples, questions)) if base_samples else None
	evaluation["decisions"] = report
	return evaluation


def generation_baseline(model, processor, spec, rows, images_dir, torch, log):
	"""The base model (adapter off) answering the same questions the usual way, by generating JSON with the
	schema in its prompt: the structured-output baseline the decision readout is compared with."""
	from .prompts import build_system_prompt
	from .trainer import answer

	schema = spec.get("schema") or decisions.questions_schema(spec["questions"])
	json_spec = {**spec, "system_prompt": build_system_prompt(spec.get("instructions", ""), schema)}
	tokenizer = processor.tokenizer if vision.is_vision(spec) else processor
	longest = max(len(tokenizer(compact_json(r["output"]), add_special_tokens=False)["input_ids"]) for r in rows)
	log(f"Generation baseline: the base model writing JSON for {len(rows)} example(s)")
	started = time.perf_counter()
	with model.disable_adapter():
		texts = answer(model, processor, json_spec, rows, images_dir, int(longest * 1.5) + 32, 4, torch)
	seconds = time.perf_counter() - started
	scored = score_predictions(rows, texts, schema)
	counts = {q["name"]: sum(1 for r in rows if isinstance(r["output"], dict) and q["name"] in r["output"]) for q in spec["questions"]}
	hits = sum(scored["field_accuracy"].get(name, 0) * n for name, n in counts.items())
	generated = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
	return {
		"accuracy": round(hits / max(1, sum(counts.values())), 4),
		"json_valid": scored["json_valid"],
		"questions": [{"name": name, "accuracy": scored["field_accuracy"].get(name, 0.0)} for name in counts],
		"latency_ms": round(seconds * 1000 / max(1, len(rows)), 1),
		"output_tokens": round(sum(generated) / max(1, len(generated)), 1),
	}


def calibration_file(report, labels=None):
	"""What serving needs from the evaluation: the temperature, each question's precision curve, and the
	training labels' distribution (the reference for drift monitoring)."""
	return {
		"temperature": report["temperature"],
		"fitted": report["calibrated"],
		"count": report["count"],
		"curves": {q["name"]: q["curve"] for q in report["questions"] if q.get("curve")},
		"labels": labels or {},
	}


class DecisionModel:
	"""A loaded decision model: its reader, the questions it was trained on, and its calibration."""

	def __init__(self, reader, name, questions=None, temperature=1.0, curves=None, base_model=None):
		self.reader = reader
		self.name = name
		self.questions = questions or []
		self.temperature = temperature
		self.curves = curves or {}
		self.base_model = base_model
		self.adapter = None  # LoRA adapter name when several runs share one base model
		self.source = None  # run or export folder
		self.labels = {}  # training label distribution per question, the reference for drift

	@property
	def vision(self):
		return self.reader.vision

	def read(self, state, questions, settings=None, image=None):
		"""(scores, usage) for questions about the input."""
		settings = settings or decisions.parse_settings({})
		image = settings.get("image") if image is None else image
		if self.adapter is not None:
			self.reader.model.set_adapter(self.adapter)
		return self.reader.score(state, questions, samples=settings["samples"], image=image)

	def answer(self, state, questions, settings=None, image=None):
		settings = settings or decisions.parse_settings({})
		scores, usage = self.read(state, questions, settings, image)
		return decisions.build_response(self.name, questions, scores, usage, self.temperature, settings, self.curves)


def predict_items(reader, spec, calibration, rows, name, images_dir=None):
	"""Try-it answers: the Decisions API response per input, plus the answer as a project output value."""
	questions = spec["questions"]
	temperature, curves = calibration
	served = DecisionModel(reader, name, questions, temperature, curves)
	items = []
	for row in rows:
		text = row["input"]
		response = served.answer(text, questions, image=row_image(row, images_dir))
		value = {}
		for q, a in zip(questions, response["answers"]):
			probs = [a["probability"], 1 - a["probability"]] if q["type"] == "predicate" else list(a["probabilities"].values())
			value[q["name"]] = decisions.predicted_value(q, probs)
		items.append({
			"input": text,
			"image": row.get("image"),
			"generated": compact_json(value),
			"value": value,
			"json_valid": True,
			"errors": validate_output(value, spec["schema"]),
			"decisions": response,
		})
	return items


def decision_source(path):
	"""(spec, adapter folder) for a decision run folder or a decision export folder, checked without PyTorch."""
	if (path / "run.json").exists():
		spec = read_json(path / "run.json")
		if not decisions.is_decision_spec(spec):
			raise decisions.DecisionError("This run answers with JSON. Serve a run trained with the Decisions answer, or use --model for a base model.")
		status = read_json(path / "status.json", {}) or {}
		if status.get("state") != "completed" or not (path / "adapter").exists():
			raise decisions.DecisionError("Only completed training runs can answer decisions.")
		return spec, path / "adapter"
	manifest = read_json(path / "llmzempic.json", None)
	if manifest is None:
		raise decisions.DecisionError(f"{path} is neither a training run nor an export folder.")
	if manifest.get("answer_mode") != "decisions":
		raise decisions.DecisionError("This export answers with JSON, not decisions.")
	contract = read_json(path / (manifest.get("decisions") or "decisions.json"))
	if not (path / "adapter").exists():
		raise decisions.DecisionError("This export has no adapter/ folder; export the run again.")
	spec = {
		"answer_mode": "decisions",
		"base_model": manifest["base_model"],
		"input_kind": manifest.get("input_kind", "text"),
		"instructions": contract.get("instructions", ""),
		"questions": [decisions.from_api_question(q) for q in contract["questions"]],
		"project_name": manifest.get("name"),
		"training": {"quantization": "auto"},
	}
	return spec, path / "adapter"


def load_model(path=None, model_id=None, name=None, shared=None):
	"""A DecisionModel for a decision run or export folder, or for any base model read training-free (no
	questions are trained; every request brings its own). shared: {base model: (PeftModel, processor)} so
	runs on the same base model share its weights and switch LoRA adapters per request."""
	import torch

	from .runs import model_name
	from .trainer import Plan, hf_token, model_kwargs
	from .workspace import catalog_entry

	if path is not None:
		from .predictor import load_trained

		spec, adapter_dir = decision_source(path)
		entry = catalog_entry(spec["base_model"])
		spec.setdefault("params_b", entry["params_b"] if entry else None)
		name = name or model_name(spec)
		adapter = None
		if shared is not None and spec["base_model"] in shared:
			model, processor = shared[spec["base_model"]]
			adapter = f"a{len(model.peft_config)}"
			model.load_adapter(str(adapter_dir), adapter_name=adapter)
		else:
			model, processor = load_trained(spec, adapter_dir, Plan(spec, torch))
			if shared is not None:
				shared[spec["base_model"]] = (model, processor)
				adapter = model.active_adapter
		served = DecisionModel(make_reader(model, processor, spec), name, spec["questions"], *run_calibration(path), spec["base_model"])
		served.adapter = adapter
		served.source = path
		served.labels = (read_json(path / "calibration.json", {}) or {}).get("labels") or {}
		return served

	from transformers import AutoModelForCausalLM, AutoTokenizer

	from .prompts import FALLBACK_CHAT_TEMPLATE
	from .workspace import catalog_entry

	# With a known size, "auto" quantization only goes 4-bit when the model wouldn't fit in 16-bit.
	entry = catalog_entry(model_id)
	is_vision = bool(entry.get("vision")) if entry else is_vision_model(model_id, hf_token())
	plan = Plan({"training": {}, "input_kind": "image" if is_vision else "text", "params_b": entry["params_b"] if entry else None}, torch)
	if is_vision:
		processor = vision.load_processor(model_id, hf_token())
		model = vision.load_model(model_id, model_kwargs(plan))
		model.eval()
		return DecisionModel(VisionDecisionReader(model, processor), name or model_id, base_model=model_id)
	tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token())
	model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs(plan))
	model.eval()
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if not getattr(tokenizer, "chat_template", None):
		tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
	return DecisionModel(DecisionReader(model, tokenizer), name or model_id, base_model=model_id)


def is_vision_model(model_id, token=None):
	try:
		vision.check_vision_model(model_id, token)
	except RuntimeError:
		return False
	return True
