"""Client code for a decision model's Decisions API (`llmzempic serve`)."""

import json

from .client_types import pascal, snake
from .decision_server import DEFAULT_PORT
from .decisions import api_question

BASE_URL = f"http://localhost:{DEFAULT_PORT}"


def request_body(name, questions, text="Text to decide on"):
	return {"model": name, "input": text, "questions": [api_question(q) for q in questions]}


def python_client(name, questions):
	names = ", ".join(q["name"] for q in questions)
	return f'''"""Decisions API client for {name}. Start the server with: llmzempic serve --run <run folder>"""

import json
import urllib.request

BASE_URL = "{BASE_URL}"


def decide(text, questions=None, min_confidence=None, base_url=BASE_URL):
	"""Answers keyed by question name ({names}). Leave questions out to ask every trained question.
	Each answer has "choice", "score" or "probability", plus "probabilities" and "confidence"."""
	body = {{"input": text}}
	if questions is not None:
		body["questions"] = questions
	if min_confidence is not None:
		body["min_confidence"] = min_confidence
	request = urllib.request.Request(
		base_url.rstrip("/") + "/v1/decisions",
		data=json.dumps(body).encode("utf-8"),
		headers={{"Content-Type": "application/json"}},
	)
	with urllib.request.urlopen(request, timeout=120) as response:
		answers = json.load(response)["answers"]
	return {{a["name"]: a for a in answers}}


if __name__ == "__main__":
	import sys

	print(json.dumps(decide(" ".join(sys.argv[1:]) or "Text to decide on"), indent=2))
'''


def php_client(name, questions):
	fn = snake(name) + "_decide"
	return f'''<?php
// Decisions API client for {name}. Start the server with: llmzempic serve --run <run folder>
// Returns the answers keyed by question name; each has choice / score / probability, probabilities and confidence.
function {fn}(string $input, ?float $minConfidence = null, string $baseUrl = '{BASE_URL}'): array
{{
	$body = ['input' => $input];
	if ($minConfidence !== null) {{
		$body['min_confidence'] = $minConfidence;
	}}
	$ch = curl_init(rtrim($baseUrl, '/') . '/v1/decisions');
	curl_setopt_array($ch, [
		CURLOPT_POST => true,
		CURLOPT_RETURNTRANSFER => true,
		CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
		CURLOPT_POSTFIELDS => json_encode($body),
	]);
	$raw = curl_exec($ch);
	$status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
	curl_close($ch);
	if ($raw === false || $status !== 200) {{
		throw new RuntimeException("Decisions API error ($status): $raw");
	}}
	$answers = [];
	foreach (json_decode($raw, true)['answers'] as $answer) {{
		$answers[$answer['name']] = $answer;
	}}
	return $answers;
}}

// print_r({fn}('Text to decide on'));
'''


def csharp_client(name, questions):
	cls = pascal(name) + "Decisions"
	return f'''// Decisions API client for {name}. Start the server with: llmzempic serve --run <run folder>
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;

public class {cls}
{{
	private readonly HttpClient http = new HttpClient();
	private readonly string url;

	public {cls}(string baseUrl = "{BASE_URL}")
	{{
		url = baseUrl.TrimEnd('/') + "/v1/decisions";
	}}

	// Answers keyed by question name; each has "choice", "score" or "probability", plus "probabilities" and "confidence".
	public async Task<Dictionary<string, JsonElement>> DecideAsync(string input, double? minConfidence = null)
	{{
		var body = new Dictionary<string, object> {{ ["input"] = input }};
		if (minConfidence is double threshold) body["min_confidence"] = threshold;
		using var response = await http.PostAsJsonAsync(url, body);
		response.EnsureSuccessStatusCode();
		var result = await response.Content.ReadFromJsonAsync<JsonElement>();
		var answers = new Dictionary<string, JsonElement>();
		foreach (var answer in result.GetProperty("answers").EnumerateArray())
			answers[answer.GetProperty("name").GetString()!] = answer;
		return answers;
	}}
}}
'''


def curl_snippet(name, questions, run_dir="<run folder>"):
	body = json.dumps(request_body(name, questions), indent=2, ensure_ascii=False)
	return (
		f'llmzempic serve --run "{run_dir}" --port {DEFAULT_PORT}\n\n'
		f"curl {BASE_URL}/v1/decisions -H \"Content-Type: application/json\" -d @- <<'JSON'\n{body}\nJSON\n"
	)


def all_decision_snippets(name, questions, run_dir="<run folder>"):
	return {
		"python": python_client(name, questions),
		"php": php_client(name, questions),
		"csharp": csharp_client(name, questions),
		"curl": curl_snippet(name, questions, run_dir),
	}


VLLM_CLIENT = '''"""Decisions for __NAME__ from a vLLM server, or any OpenAI-compatible completions server that returns prompt
log-probabilities with echo. Exact: each option number is scored as part of the prompt, so no text is generated.

	vllm serve <export folder>/merged --served-model-name __NAME__
	vllm serve __BASE__ --enable-lora --lora-modules __NAME__=<export folder>/adapter
"""

import json
import math
import urllib.request

BASE_URL = "http://localhost:8000"
MODEL = "__NAME__"
CONTRACT = json.loads(__CONTRACT__)


def _post(url, body):
	request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
	with urllib.request.urlopen(request, timeout=120) as response:
		return json.load(response)


def _prompt(question, text):
	template = CONTRACT["prompt_template"]
	head, _, tail = question["prompt"].partition("{input}")
	return template["before"] + head + text + tail + template["after"]


def _answer(question, scores, min_confidence):
	t = CONTRACT["temperature"]
	top = max(s / t for s in scores)
	exps = [math.exp(s / t - top) for s in scores]
	probs = [e / sum(exps) for e in exps]
	best = max(range(len(probs)), key=probs.__getitem__)
	refused = min_confidence is not None and probs[best] < min_confidence
	answer = {"name": question["name"], "type": question["type"], "confidence": round(probs[best], 4)}
	if question["type"] == "predicate":
		answer["probability"] = round(probs[0], 4)
	else:
		labels = question.get("choices") or question.get("levels")
		if question["type"] == "choice":
			answer["choice"] = None if refused else labels[best]
		else:
			answer["score"] = None if refused else round(sum(p * (i + 1) for i, p in enumerate(probs)), 4)
			answer["level"] = None if refused else labels[best]
		answer["probabilities"] = {label: round(p, 4) for label, p in zip(labels, probs)}
	if refused:
		answer["refused"] = True
	return answer


def decide(text, names=None, min_confidence=None, base_url=BASE_URL, model=MODEL):
	"""Answers keyed by question name, like the Decisions API. min_confidence "auto" uses each question's
	threshold for 90% held-out accuracy."""
	answers = {}
	for question in CONTRACT["questions"]:
		if names and question["name"] not in names:
			continue
		prompt = _prompt(question, text)
		# The template's answer prefix (a space for some models) comes before every number. It's scored
		# with the number: as its own token it adds the same to every option, merged into the number it must count.
		numbers = [CONTRACT["prompt_template"].get("answer_prefix", "") + n for n in question["numbers"]]
		body = {"model": model, "prompt": [prompt + n for n in numbers], "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1}
		scores = [0.0] * len(numbers)
		for choice in _post(base_url.rstrip("/") + "/v1/completions", body)["choices"]:
			i = choice["index"]
			logprobs = choice["logprobs"]
			end = len(prompt) + len(numbers[i])
			scores[i] = sum(lp for lp, at in zip(logprobs["token_logprobs"], logprobs["text_offset"]) if lp is not None and len(prompt) <= at < end)
		threshold = question.get("auto_min_confidence") if min_confidence == "auto" else min_confidence
		answers[question["name"]] = _answer(question, scores, threshold)
	return answers


if __name__ == "__main__":
	import sys

	print(json.dumps(decide(" ".join(sys.argv[1:]) or "Text to decide on"), indent=2))
'''


def vllm_client(contract):
	"""A dependency-free Python client that reads a decision model's probabilities from vLLM."""
	return (
		VLLM_CLIENT.replace("__NAME__", contract["name"]).replace("__BASE__", contract["base_model"])
		.replace("__CONTRACT__", repr(json.dumps(contract, ensure_ascii=False)))
	)
