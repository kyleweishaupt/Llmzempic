"""Synthetic examples from a stronger LLM behind any OpenAI-compatible chat endpoint (Ollama, llama-server,
vLLM, OpenAI). For each value of each option list or yes/no field it asks for varied, realistic inputs with
that answer, including hard cases that look like another value, together with the answers to the other
fields. Every example is checked against the schema; ones that don't match are dropped.
"""

import json
import threading
import time
import urllib.error
import urllib.request
import uuid

from .dataset import input_key, label_fields
from .metrics import parse_generated
from .schema_tools import compact_json, validate_output

MAX_PER_VALUE = 50


class SynthesisError(Exception):
	pass


def chat(settings, messages, timeout=300):
	"""The assistant's reply from POST {generator_url}/chat/completions."""
	url = str(settings.get("generator_url") or "").rstrip("/")
	if not url or not settings.get("generator_model"):
		raise SynthesisError("Set the generator's URL and model in Settings first.")
	body = {"model": settings["generator_model"], "messages": messages, "temperature": 0.9}
	headers = {"Content-Type": "application/json"}
	if settings.get("generator_key"):
		headers["Authorization"] = f"Bearer {settings['generator_key']}"
	request = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode("utf-8"), headers=headers)
	try:
		with urllib.request.urlopen(request, timeout=timeout) as response:
			return json.load(response)["choices"][0]["message"]["content"]
	except urllib.error.HTTPError as e:
		raise SynthesisError(f"The generator answered {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
	except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as e:
		raise SynthesisError(f"Couldn't get examples from the generator: {e}")


def targets(schema, fields=None):
	"""(field, value) pairs to generate for: every value of every option list and yes/no field."""
	out = []
	for name, kind, values in label_fields(schema):
		if kind != "tags" and (not fields or name in fields):
			out += [(name, v) for v in values]
	return out


def prompt_messages(instructions, schema, field, value, count, others, samples):
	shown = ", ".join(json.dumps(v) for v in others)
	lines = [
		f"Write {count} new, varied and realistic inputs for this task, each with its correct answer.",
		f"Task: {instructions.strip() or 'Answer about the input.'}",
		f"Answers are JSON objects matching this JSON Schema: {compact_json(schema)}",
		f'For every input the correct value of "{field}" must be {json.dumps(value)}.',
		f"Make about a third of them hard cases that could be mistaken for {shown}." if others else "",
		"Vary length, tone, wording and topic. Don't repeat the examples.",
		'Reply with only JSON: {"examples": [{"input": "...", "output": {...}}, ...]}',
	]
	if samples:
		lines.append("Existing examples, for style:\n" + "\n".join(compact_json(s) for s in samples))
	return [
		{"role": "system", "content": "You write labeled training examples for a classifier. You reply with JSON only."},
		{"role": "user", "content": "\n".join(line for line in lines if line)},
	]


def parse_examples(text, schema, field, value):
	"""Examples from the generator's reply that match the schema and have the asked-for value."""
	data, ok = parse_generated(text)
	if not ok:
		return []
	items = data.get("examples") if isinstance(data, dict) else data
	out = []
	for item in items if isinstance(items, list) else []:
		if not isinstance(item, dict) or not isinstance(item.get("input"), str) or not item["input"].strip():
			continue
		output = item.get("output")
		if not isinstance(output, dict) or output.get(field) != value or validate_output(output, schema):
			continue
		out.append({"input": item["input"].strip(), "output": output})
	return out


def generate(settings, instructions, schema, per_value, samples=(), fields=None, on_batch=None, stopped=lambda: False, ask=chat):
	"""Generate per_value examples for every (field, value); on_batch(examples, done, total) after each."""
	per_value = max(1, min(int(per_value), MAX_PER_VALUE))
	pairs = targets(schema, fields)
	if not pairs:
		raise SynthesisError("Synthetic examples need an option list or a yes/no field in the output.")
	seen = set()
	for done, (field, value) in enumerate(pairs, start=1):
		if stopped():
			return
		others = [v for f, v in pairs if f == field and v != value]
		reply = ask(settings, prompt_messages(instructions, schema, field, value, per_value, others, list(samples)[:3]))
		batch = []
		for example in parse_examples(reply, schema, field, value):
			key = input_key(example)
			if key not in seen:
				seen.add(key)
				batch.append({"id": f"syn-{uuid.uuid4().hex[:10]}", **example})
		if on_batch:
			on_batch(batch[:per_value], done, len(pairs))


class SynthesisJobs:
	"""One background generation per project, with progress for the UI."""

	def __init__(self):
		self.lock = threading.Lock()
		self.jobs = {}

	def status(self, project_id):
		with self.lock:
			return dict(self.jobs.get(project_id) or {"running": False})

	def stop(self, project_id):
		with self.lock:
			if project_id in self.jobs:
				self.jobs[project_id]["stop"] = True

	def start(self, project_id, run):
		"""run(update, stopped) does the work; update(**fields) reports progress."""
		with self.lock:
			if (self.jobs.get(project_id) or {}).get("running"):
				raise SynthesisError("Examples are already being generated for this project.")
			self.jobs[project_id] = {"running": True, "done": 0, "total": 0, "added": 0, "error": None, "started": time.time(), "stop": False}

		def update(**fields):
			with self.lock:
				self.jobs[project_id].update(fields)

		def work():
			try:
				run(update, lambda: self.jobs[project_id]["stop"])
			except Exception as e:
				update(error=str(e))
			finally:
				update(running=False)

		threading.Thread(target=work, daemon=True).start()
