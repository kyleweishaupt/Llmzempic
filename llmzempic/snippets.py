"""Ready-to-paste client code for calling an exported model from other projects.

Every client asks the server to constrain decoding to the JSON Schema (Ollama `format`,
llama-server `response_format`), so responses always parse; the fine-tune makes them correct.
Data set projects get llama-server clients instead (server_clients.py): the server loads a grammar
that holds every data set field to its list. Clients for image models take an image file (sent
base64-encoded) plus optional text.
"""

import json
import pprint

from .client_types import cs_result, pascal, snake
from .schema_tools import compact_json
from .server_clients import csharp_server_client, php_server_client, python_server_client


def php_client(name, schema, vision=False):
	fn = snake(name)
	const = fn.upper() + "_SCHEMA"
	if vision:
		params = "string $imagePath, string $text = ''"
		read = (
			"\t$image = file_get_contents($imagePath);\n"
			"\tif ($image === false) {\n"
			"\t\tthrow new RuntimeException(\"Could not read $imagePath\");\n"
			"\t}\n"
		)
		message = "['role' => 'user', 'content' => $text, 'images' => [base64_encode($image)]]"
		example = f"{fn}('photo.jpg')"
	else:
		params, read = "string $input", ""
		message = "['role' => 'user', 'content' => $input]"
		example = f"{fn}('Text to analyze')"
	return f"""<?php
// Calls the fine-tuned "{name}" model through Ollama (https://ollama.com).
// One-time setup, from the export folder: ollama create {name} -f Modelfile
// The instructions and schema are baked into the model's system prompt; `format`
// additionally forces the reply to match the schema.

const {const} = <<<'JSON'
{json.dumps(schema, indent=4, ensure_ascii=False)}
JSON;

function {fn}({params}, string $host = 'http://localhost:11434'): array
{{
{read}	$payload = [
		'model' => '{name}',
		'messages' => [{message}],
		'format' => json_decode({const}, true),
		'stream' => false,
		'options' => ['temperature' => 0],
	];
	$ch = curl_init(rtrim($host, '/') . '/api/chat');
	curl_setopt_array($ch, [
		CURLOPT_POST => true,
		CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
		CURLOPT_POSTFIELDS => json_encode($payload, JSON_UNESCAPED_UNICODE),
		CURLOPT_RETURNTRANSFER => true,
		CURLOPT_TIMEOUT => 120,
	]);
	$body = curl_exec($ch);
	if ($body === false) {{
		throw new RuntimeException('Model request failed: ' . curl_error($ch));
	}}
	$status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
	if ($status !== 200) {{
		throw new RuntimeException("Model server returned HTTP $status: $body");
	}}
	$response = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
	return json_decode($response['message']['content'], true, 512, JSON_THROW_ON_ERROR);
}}

// Example:
// $result = {example};
// echo $result[array_key_first($result)];
"""


def csharp_client(name, schema, vision=False):
	cls = pascal(name)
	if vision:
		params = 'string imagePath, string input = "", CancellationToken cancellationToken = default'
		read = "var image = Convert.ToBase64String(await File.ReadAllBytesAsync(imagePath, cancellationToken));\n\t\t"
		message = '["role"] = "user", ["content"] = input, ["images"] = new JsonArray(image)'
	else:
		params = "string input, CancellationToken cancellationToken = default"
		read = ""
		message = '["role"] = "user", ["content"] = input'
	result_type, classes, deserialize = cs_result(schema, cls)
	schema_literal = compact_json(schema).replace('"', '""')
	return f"""// Calls the fine-tuned "{name}" model through Ollama (https://ollama.com). Targets .NET 6+.
// One-time setup, from the export folder: ollama create {name} -f Modelfile
// To use llama-server instead, POST to /v1/chat/completions with the system prompt from
// system_prompt.txt and response_format = {{ type: "json_schema", json_schema: {{ schema }} }}.
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

public sealed class {cls}Client
{{
	private const string Schema = @"{schema_literal}";
	private static readonly HttpClient SharedHttp = new() {{ Timeout = TimeSpan.FromMinutes(2) }};

	private readonly HttpClient _http;
	private readonly string _baseUrl;
	private readonly string _model;

	public {cls}Client(string baseUrl = "http://localhost:11434", string model = "{name}", HttpClient? http = null)
	{{
		_http = http ?? SharedHttp;
		_baseUrl = baseUrl.TrimEnd('/');
		_model = model;
	}}

	public async Task<{result_type}> RunAsync({params})
	{{
		{read}var payload = new JsonObject
		{{
			["model"] = _model,
			["messages"] = new JsonArray(new JsonObject {{ {message} }}),
			["format"] = JsonNode.Parse(Schema),
			["stream"] = false,
			["options"] = new JsonObject {{ ["temperature"] = 0 }},
		}};
		using var content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
		using var response = await _http.PostAsync(_baseUrl + "/api/chat", content, cancellationToken);
		response.EnsureSuccessStatusCode();
		using var reply = JsonDocument.Parse(await response.Content.ReadAsStringAsync(cancellationToken));
		var text = reply.RootElement.GetProperty("message").GetProperty("content").GetString() ?? "null";
		{deserialize}
	}}
}}

{classes}
"""


def python_client(name, schema, vision=False):
	fn = snake(name)
	if vision:
		imports = "import base64\nimport json\nimport urllib.request"
		params = 'image_path, text=""'
		read = '\twith open(image_path, "rb") as f:\n\t\timage = base64.b64encode(f.read()).decode("ascii")\n'
		message = '{"role": "user", "content": text, "images": [image]}'
		example = f'{fn}("photo.jpg")'
	else:
		imports = "import json\nimport urllib.request"
		params, read = "text", ""
		message = '{"role": "user", "content": text}'
		example = f'{fn}("Text to analyze")'
	return f'''"""Calls the fine-tuned "{name}" model through Ollama. Setup: ollama create {name} -f Modelfile"""
{imports}

SCHEMA = {pprint.pformat(schema, indent=1, width=100, sort_dicts=False)}

def {fn}({params}, host="http://localhost:11434"):
{read}	payload = {{
		"model": "{name}",
		"messages": [{message}],
		"format": SCHEMA,
		"stream": False,
		"options": {{"temperature": 0}},
	}}
	request = urllib.request.Request(
		host.rstrip("/") + "/api/chat",
		data=json.dumps(payload).encode("utf-8"),
		headers={{"Content-Type": "application/json"}},
	)
	with urllib.request.urlopen(request, timeout=120) as response:
		reply = json.load(response)
	return json.loads(reply["message"]["content"])


if __name__ == "__main__":
	print({example})
'''


IMAGE_PLACEHOLDER = "data:image/jpeg;base64,PUT_BASE64_IMAGE_HERE"


def llama_server_request(system_prompt, schema, example_input="Text to analyze", vision=False):
	"""Without a schema (data set projects) the server's --grammar-file does the constraining: a
	response_format would make it build a grammar for every request instead."""
	content = example_input
	if vision:
		# OpenAI-style content parts; the text part is optional, like in training.
		content = [{"type": "image_url", "image_url": {"url": IMAGE_PLACEHOLDER}}]
		if example_input:
			content.append({"type": "text", "text": example_input})
	request = {
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": content},
		],
		"temperature": 0,
	}
	if schema is not None:
		request["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}
	return request


def llama_server_command(gguf_file, ctx, mmproj_file=None, image_tokens=None, grammar=False):
	images = ""
	if mmproj_file:
		images = f" --mmproj {mmproj_file}" + (f" --image-max-tokens {image_tokens}" if image_tokens else "")
	return f"llama-server -m {gguf_file or 'model.gguf'}{images}{' --grammar-file grammar.gbnf' if grammar else ''} --port 8080 -c {ctx}"


def llama_server_usage(gguf_file, ctx, mmproj_file=None, image_tokens=None, grammar=False):
	usage = (
		f"# Serve the model with llama.cpp (https://github.com/ggml-org/llama.cpp/releases)\n"
		+ ("# grammar.gbnf keeps every data set field to its list (needs a 2024 or later llama-server).\n" if grammar else "")
		+ llama_server_command(gguf_file, ctx, mmproj_file, image_tokens, grammar) + "\n\n"
		"# Call it (OpenAI-compatible). request.json includes the system prompt the model was trained with.\n"
	)
	if mmproj_file:
		usage += f"# Replace {IMAGE_PLACEHOLDER.split(',')[1]} with the image file's base64 (data:image/png;base64,... for PNG).\n"
	return usage + "curl http://localhost:8080/v1/chat/completions -H \"Content-Type: application/json\" -d @llama-server-request.json\n"


def all_snippets(name, schema, system_prompt, gguf_file=None, ctx=4096, vision=False, mmproj_file=None, image_tokens=None, refs=None):
	example = "" if vision else "Text to analyze"
	if vision and gguf_file and not mmproj_file:
		mmproj_file = f"mmproj-{name}-f16.gguf"
	request = llama_server_request(system_prompt, None if refs else schema, example, vision)
	if refs:
		serve = llama_server_command(gguf_file, ctx, mmproj_file if vision else None, image_tokens, True)
		clients = {
			"php": php_server_client(name, system_prompt, vision, refs, serve),
			"csharp": csharp_server_client(name, schema, system_prompt, vision, refs, serve),
			"python": python_server_client(name, system_prompt, vision, refs, serve),
		}
	else:
		clients = {"php": php_client(name, schema, vision), "csharp": csharp_client(name, schema, vision), "python": python_client(name, schema, vision)}
	return {
		**clients,
		"llama_server": llama_server_usage(gguf_file, ctx, mmproj_file if vision else None, image_tokens, bool(refs))
		+ "\n# llama-server-request.json\n"
		+ json.dumps(request, indent=2, ensure_ascii=False),
	}
