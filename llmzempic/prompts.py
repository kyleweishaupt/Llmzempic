"""Prompt construction and chat-template tokenization.

The exact system prompt built here is baked into every export (Modelfile, manifest) so the
deployed model sees the same text it was trained on.
"""

from .schema_tools import compact_json

SCHEMA_HEADER = "Respond with only a JSON value that conforms to this JSON Schema, with no other text:"

# Used only for base models that ship without a chat template.
FALLBACK_CHAT_TEMPLATE = (
	"{% for message in messages %}"
	"### {{ message['role'] | capitalize }}:\n{{ message['content'] }}"
	"{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}\n\n"
	"{% endfor %}"
	"{% if add_generation_prompt %}### Assistant:\n{% endif %}"
)


def build_system_prompt(instructions, schema, include_schema=True):
	parts = [(instructions or "").strip()]
	if include_schema and schema:
		parts.append(f"{SCHEMA_HEADER}\n{compact_json(schema)}")
	return "\n\n".join(p for p in parts if p)


def format_output(output):
	"""Serialize an expected output exactly as the model should learn to write it."""
	return compact_json(output)


def build_messages(system_prompt, user_input, assistant_text=None):
	messages = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	messages.append({"role": "user", "content": user_input})
	if assistant_text is not None:
		messages.append({"role": "assistant", "content": assistant_text})
	return messages


def _merge_system(messages):
	"""Fold the system prompt into the first user turn for templates without a system role."""
	if not messages or messages[0]["role"] != "system":
		return messages
	system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
	for m in rest:
		if m["role"] == "user":
			m["content"] = f"{system}\n\n{m['content']}"
			break
	return rest


def render_chat(tokenizer, messages, add_generation_prompt):
	kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt, "enable_thinking": False}
	try:
		return tokenizer.apply_chat_template(messages, **kwargs)
	except Exception:
		if messages and messages[0]["role"] == "system":
			return tokenizer.apply_chat_template(_merge_system(messages), **kwargs)
		raise


def tokenize_example(tokenizer, system_prompt, user_input, output):
	"""Tokenize one example; loss is computed only on the assistant's answer tokens."""
	answer = format_output(output)
	prompt_text = render_chat(tokenizer, build_messages(system_prompt, user_input), True)
	full_text = render_chat(tokenizer, build_messages(system_prompt, user_input, answer), False)
	if full_text.startswith(prompt_text):
		completion_text = full_text[len(prompt_text):]
	else:
		completion_text = answer + (tokenizer.eos_token or "")
	prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
	completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
	input_ids = list(prompt_ids) + list(completion_ids)
	return {
		"input_ids": input_ids,
		"attention_mask": [1] * len(input_ids),
		"labels": [-100] * len(prompt_ids) + list(completion_ids),
	}
