from llmzempic.prompts import SCHEMA_HEADER, build_system_prompt, render_chat, tokenize_example


class FakeTokenizer:
	"""Character-level tokenizer with a ChatML-style template."""

	eos_token = "<eos>"

	def __init__(self, allow_system=True):
		self.allow_system = allow_system

	def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
		if not self.allow_system and any(m["role"] == "system" for m in messages):
			raise ValueError("System role not supported")
		text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
		if add_generation_prompt:
			text += "<|assistant|>"
		return text

	def __call__(self, text, add_special_tokens=False):
		return {"input_ids": [ord(c) for c in text]}


def decode(ids):
	return "".join(chr(i) for i in ids)


def test_system_prompt_contains_compact_schema():
	prompt = build_system_prompt("Classify it.", {"type": "object"})
	assert prompt == f'Classify it.\n\n{SCHEMA_HEADER}\n{{"type":"object"}}'
	assert build_system_prompt("Classify it.", {"type": "object"}, include_schema=False) == "Classify it."


def test_labels_cover_only_the_answer():
	tok = FakeTokenizer()
	features = tokenize_example(tok, "SYS", "hello", {"label": "a"})
	labeled = decode([i for i, label in zip(features["input_ids"], features["labels"]) if label != -100])
	assert labeled == '{"label":"a"}<|end|>'
	assert decode(features["input_ids"]).startswith("<|system|>SYS<|end|><|user|>hello")
	assert len(features["attention_mask"]) == len(features["input_ids"])


def test_system_prompt_is_merged_when_template_rejects_it():
	tok = FakeTokenizer(allow_system=False)
	text = render_chat(tok, [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}], True)
	assert text == "<|user|>SYS\n\nhi<|end|><|assistant|>"
