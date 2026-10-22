"""Image-input (vision-language) fine-tuning helpers. Runs inside the worker process.

Each image is capped to a few hundred tokens, and the capped processor is saved next to the
adapter so evaluation, Try it and export all see the resolution the model was trained on.
"""

import inspect
from pathlib import Path

from .prompts import format_output

# Qwen-VL style processors: total pixels per image (~256 tokens at 32 px per token).
MAX_IMAGE_PIXELS = 512 * 512
# Linear layers outside the language model; LoRA leaves the vision side frozen.
VISION_MODULES = r".*(vision|visual|connector|projector|merger).*"


def is_vision(spec):
	return spec.get("input_kind") == "image"


def load_image(path):
	from PIL import Image, ImageOps

	with Image.open(path) as image:
		# Phone photos store their rotation in EXIF; browsers apply it, PIL doesn't.
		return ImageOps.exif_transpose(image).convert("RGB")


def limit_resolution(processor):
	ip = processor.image_processor
	if hasattr(ip, "do_image_splitting"):
		# Idefics3 / SmolVLM: one 512 px view instead of up to 16 tiles.
		ip.do_image_splitting = False
		ip.size = {"longest_edge": ip.max_image_size["longest_edge"]}
	elif hasattr(ip, "merge_size"):
		ip.size = {"shortest_edge": min(ip.size["shortest_edge"], MAX_IMAGE_PIXELS), "longest_edge": MAX_IMAGE_PIXELS}


def image_max_tokens(processor):
	"""Most tokens one image becomes, for llama-server's --image-max-tokens."""
	ip = processor.image_processor
	if hasattr(ip, "merge_size"):
		return int(ip.size["longest_edge"] // (ip.patch_size * ip.merge_size) ** 2)
	return int(getattr(processor, "image_seq_len", 0) or 0) or None


def load_processor(source, token=None):
	from transformers import AutoProcessor

	processor = AutoProcessor.from_pretrained(str(source), token=token)
	limit_resolution(processor)
	if processor.tokenizer.pad_token is None:
		processor.tokenizer.pad_token = processor.tokenizer.eos_token
	return processor


def check_vision_model(model_id, token=None):
	from transformers import AutoConfig
	from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

	config = AutoConfig.from_pretrained(model_id, token=token)
	if config.model_type not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
		raise RuntimeError(f"{model_id} can't read images; pick a vision model.")


def load_model(model_id, kwargs):
	from transformers import AutoModelForImageTextToText

	check_vision_model(model_id, kwargs.get("token"))
	return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def build_messages(system_prompt, text, answer=None):
	user = [{"type": "image"}]
	if text.strip():
		user.append({"type": "text", "text": text})
	messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] if system_prompt else []
	messages.append({"role": "user", "content": user})
	if answer is not None:
		messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
	return messages


def render(processor, messages, add_generation_prompt):
	try:
		return processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
	except Exception:
		if messages[0]["role"] != "system":
			raise
		# Templates without a system role: fold the instructions into the user turn.
		system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
		rest[0]["content"] = rest[0]["content"][:1] + system + rest[0]["content"][1:]
		return processor.apply_chat_template(rest, add_generation_prompt=add_generation_prompt, tokenize=False)


def build_features(processor, system_prompt, rows, images_dir, log):
	"""Chat text plus the prompt length per example; the collator processes the images per batch."""
	features, warned = [], False
	for row in rows:
		path = Path(images_dir) / row["image"]
		image = load_image(path)
		prompt = render(processor, build_messages(system_prompt, row.get("input") or ""), True)
		full = render(processor, build_messages(system_prompt, row.get("input") or "", format_output(row["output"])), False)
		if not full.startswith(prompt):
			full = prompt + format_output(row["output"]) + (processor.tokenizer.eos_token or "")
		prompt_ids = processor(text=[prompt], images=[[image]], return_tensors="pt")["input_ids"][0]
		full_ids = processor(text=[full], images=[[image]], return_tensors="pt")["input_ids"][0]
		if not warned and not bool((full_ids[:len(prompt_ids)] == prompt_ids).all()):
			log("WARNING: the chat template tokenizes the prompt differently inside the full example; the loss mask may be off by a few tokens.")
			warned = True
		features.append({"text": full, "image": str(path), "prompt_len": len(prompt_ids), "length": len(full_ids)})
	return features


class VisionCollator:
	"""Processes text + images per batch (right padding) and masks everything but the answer."""

	def __init__(self, processor, model):
		self.processor = processor
		forward = getattr(model, "get_base_model", lambda: model)().forward
		self.keys = set(inspect.signature(forward).parameters)

	def __call__(self, features):
		self.processor.tokenizer.padding_side = "right"
		batch = self.processor(
			text=[f["text"] for f in features],
			images=[[load_image(f["image"])] for f in features],
			padding=True,
			return_tensors="pt",
		)
		labels = batch["input_ids"].clone()
		labels[batch["attention_mask"] == 0] = -100
		for i, f in enumerate(features):
			labels[i, :f["prompt_len"]] = -100
		batch["labels"] = labels
		return {k: v for k, v in batch.items() if k in self.keys}


def generate(model, processor, system_prompt, rows, images_dir, max_new_tokens, batch_size, torch):
	"""Greedy answers for rows of {"input", "image"}."""
	processor.tokenizer.padding_side = "left"
	gen_config = getattr(model, "generation_config", None)
	if gen_config is not None:
		gen_config.do_sample = False
		for attr in ("temperature", "top_p", "top_k"):
			setattr(gen_config, attr, None)
	texts = []
	for i in range(0, len(rows), batch_size):
		chunk = rows[i:i + batch_size]
		enc = processor(
			text=[render(processor, build_messages(system_prompt, r.get("input") or ""), True) for r in chunk],
			images=[[load_image(Path(images_dir) / r["image"])] for r in chunk],
			padding=True,
			return_tensors="pt",
		).to(model.device)
		with torch.no_grad():
			out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
		texts += processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
	return texts
