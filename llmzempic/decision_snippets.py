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
