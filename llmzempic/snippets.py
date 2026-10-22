"""Ready-to-paste client code for calling an exported model from other projects.

Every client asks the server to constrain decoding to the JSON Schema (Ollama `format`,
llama-server `response_format`), so responses always parse; the fine-tune makes them correct.
Clients for image models take an image file (sent base64-encoded) plus optional text.
"""

import json
import pprint
import re

from .schema_tools import compact_json


def pascal(name):
	words = re.split(r"[^A-Za-z0-9]+", name)
	ident = "".join(w[:1].upper() + w[1:] for w in words if w)
	return ident if ident[:1].isalpha() else f"Model{ident}"


def snake(name):
	ident = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
	return ident if ident[:1].isalpha() else f"model_{ident}"


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


_CS_SIMPLE = {"string": "string", "integer": "long", "number": "double", "boolean": "bool"}


def _cs_type(schema, name_hint, classes):
	"""Map a JSON Schema node to a C# type, generating classes for nested objects."""
	types = schema.get("type")
	types = types if isinstance(types, list) else [types]
	nullable = "null" in types
	main = next((t for t in types if t and t != "null"), None)
	if main is None and "enum" in schema:
		main = "string"
	if main == "object" and isinstance(schema.get("properties"), dict):
		return _cs_class(schema, name_hint, classes) + ("?" if nullable else ""), "" if nullable else " = new();"
	if main == "array":
		item = _cs_type(schema.get("items") or {}, name_hint + "Item", classes)[0]
		return f"List<{item}>" + ("?" if nullable else ""), "" if nullable else " = new();"
	if main in _CS_SIMPLE:
		cs = _CS_SIMPLE[main]
		if nullable:
			return cs + "?", ""
		return cs, ' = "";' if cs == "string" else ""
	return "JsonElement", ""


def _cs_class(schema, name, classes):
	base, n = name, 2
	while name in classes:
		name, n = f"{base}{n}", n + 1
	classes[name] = ""
	lines = [f"public sealed class {name}", "{"]
	for key, prop in schema["properties"].items():
		prop = prop if isinstance(prop, dict) else {}
		member = pascal(key)
		if member == name:
			member += "Value"
		cs_type, init = _cs_type(prop, member, classes)
		if prop.get("enum"):
			values = ", ".join(json.dumps(v) for v in prop["enum"])
			lines.append(f"\t/// <summary>One of: {values}</summary>")
		lines.append(f'\t[JsonPropertyName("{key}")]')
		lines.append(f"\tpublic {cs_type} {member} {{ get; set; }}{init}")
	lines.append("}")
	classes[name] = "\n".join(lines)
	return name


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
	classes = {}
	result_type = "JsonElement"
	if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
		result_type = _cs_class(schema, f"{cls}Result", classes)
	schema_literal = compact_json(schema).replace('"', '""')
	deserialize = (
		"using var result = JsonDocument.Parse(text);\n\t\treturn result.RootElement.Clone();"
		if result_type == "JsonElement"
		else f"return JsonSerializer.Deserialize<{result_type}>(text)\n\t\t\t?? throw new InvalidOperationException(\"Model returned null.\");"
	)
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

{(chr(10) + chr(10)).join(classes.values())}
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
	content = example_input
	if vision:
		# OpenAI-style content parts; the text part is optional, like in training.
		content = [{"type": "image_url", "image_url": {"url": IMAGE_PLACEHOLDER}}]
		if example_input:
			content.append({"type": "text", "text": example_input})
	return {
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": content},
		],
		"temperature": 0,
		"response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}},
	}


def llama_server_usage(gguf_file, ctx, mmproj_file=None, image_tokens=None):
	images = ""
	if mmproj_file:
		images = f" --mmproj {mmproj_file}" + (f" --image-max-tokens {image_tokens}" if image_tokens else "")
	usage = (
		f"# Serve the model with llama.cpp (https://github.com/ggml-org/llama.cpp/releases)\n"
		f"llama-server -m {gguf_file or 'model.gguf'}{images} --port 8080 -c {ctx}\n\n"
		"# Call it (OpenAI-compatible). request.json includes the system prompt the model was trained with.\n"
	)
	if mmproj_file:
		usage += f"# Replace {IMAGE_PLACEHOLDER.split(',')[1]} with the image file's base64 (data:image/png;base64,... for PNG).\n"
	return usage + "curl http://localhost:8080/v1/chat/completions -H \"Content-Type: application/json\" -d @llama-server-request.json\n"


def all_snippets(name, schema, system_prompt, gguf_file=None, ctx=4096, vision=False, mmproj_file=None, image_tokens=None):
	example = "" if vision else "Text to analyze"
	if vision and gguf_file and not mmproj_file:
		mmproj_file = f"mmproj-{name}-f16.gguf"
	return {
		"php": php_client(name, schema, vision),
		"csharp": csharp_client(name, schema, vision),
		"python": python_client(name, schema, vision),
		"llama_server": llama_server_usage(gguf_file, ctx, mmproj_file if vision else None, image_tokens)
		+ "\n# llama-server-request.json\n"
		+ json.dumps(llama_server_request(system_prompt, schema, example, vision), indent=2, ensure_ascii=False),
	}
