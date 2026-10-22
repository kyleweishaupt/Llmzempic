"""Clients for data set projects. They call llama.cpp's llama-server, which loads grammar.gbnf at startup
and so only answers with values from the data sets; Ollama can't enforce lists that long. An opt-in check
re-reads answers against the lists in datasets/, for Ollama or a server started without the grammar."""

from .client_checks import csharp_checks, php_checks, python_checks
from .client_types import cs_result, pascal, snake

PROMPT_NOTE = "The exact prompt the model was trained with; llama-server has no Modelfile to hold it."


def php_server_client(name, system_prompt, vision, refs, serve):
	fn = snake(name)
	if vision:
		params = "string $imagePath, string $text = ''"
		read = (
			"\t$image = file_get_contents($imagePath);\n"
			"\tif ($image === false) {\n"
			"\t\tthrow new RuntimeException(\"Could not read $imagePath\");\n"
			"\t}\n"
			"\t$media = ['png' => 'image/png', 'webp' => 'image/webp', 'gif' => 'image/gif'][strtolower(pathinfo($imagePath, PATHINFO_EXTENSION))] ?? 'image/jpeg';\n"
			"\t$content = [['type' => 'image_url', 'image_url' => ['url' => \"data:$media;base64,\" . base64_encode($image)]]];\n"
			"\tif ($text !== '') {\n"
			"\t\t$content[] = ['type' => 'text', 'text' => $text];\n"
			"\t}\n"
		)
		user, example = "$content", f"{fn}('photo.jpg')"
	else:
		params, read, user, example = "string $input", "", "$input", f"{fn}('Text to analyze')"
	prompt = fn.upper() + "_SYSTEM_PROMPT"
	return f"""<?php
// Calls the fine-tuned "{name}" model through llama.cpp's llama-server. Start it from the export folder:
//   {serve}
// With grammar.gbnf every data set field answers from its list. $checkValues = true re-checks answers
// against ../datasets (keep that folder next to this file), e.g. with Ollama or a server started without it.

// {PROMPT_NOTE}
const {prompt} = <<<'LLMZEMPIC_PROMPT'
{system_prompt}
LLMZEMPIC_PROMPT;

function {fn}({params}, string $host = 'http://localhost:8080', bool $checkValues = false): array
{{
{read}	$payload = [
		'messages' => [
			['role' => 'system', 'content' => {prompt}],
			['role' => 'user', 'content' => {user}],
		],
		'temperature' => 0,
	];
	$ch = curl_init(rtrim($host, '/') . '/v1/chat/completions');
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
	$result = json_decode($response['choices'][0]['message']['content'], true, 512, JSON_THROW_ON_ERROR);
	return $checkValues ? {fn}_check_values($result) : $result;
}}
{php_checks(fn, refs)}
// Example:
// $result = {example};
// echo json_encode($result);
"""


def csharp_server_client(name, schema, system_prompt, vision, refs, serve):
	cls = pascal(name)
	result_type, classes, deserialize = cs_result(schema, cls)
	if vision:
		params = 'string imagePath, string input = "", CancellationToken cancellationToken = default'
		read = (
			"var bytes = await File.ReadAllBytesAsync(imagePath, cancellationToken);\n"
			"\t\tvar media = Path.GetExtension(imagePath).ToLowerInvariant() switch\n"
			"\t\t{\n"
			'\t\t\t".png" => "image/png",\n'
			'\t\t\t".webp" => "image/webp",\n'
			'\t\t\t".gif" => "image/gif",\n'
			'\t\t\t_ => "image/jpeg",\n'
			"\t\t};\n"
			'\t\tvar user = new JsonArray(new JsonObject { ["type"] = "image_url", ["image_url"] = new JsonObject { ["url"] = $"data:{media};base64,{Convert.ToBase64String(bytes)}" } });\n'
			"\t\tif (input.Length > 0)\n"
			"\t\t{\n"
			'\t\t\tuser.Add(new JsonObject { ["type"] = "text", ["text"] = input });\n'
			"\t\t}\n"
			"\t\t"
		)
		user = "user"
	else:
		params = "string input, CancellationToken cancellationToken = default"
		read, user = "", "input"
	return f"""// Calls the fine-tuned "{name}" model through llama.cpp's llama-server. Targets .NET 6+.
// Start it from the export folder:
//   {serve}
// With grammar.gbnf every data set field answers from its list. checkValues: true re-checks answers
// against the export's datasets/ folder (copy it next to your app, or pass its parent folder as
// exportDir), e.g. with Ollama or a server started without the grammar.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
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
	// {PROMPT_NOTE}
	private const string SystemPrompt = @"{system_prompt.replace('"', '""')}";
	private static readonly HttpClient SharedHttp = new() {{ Timeout = TimeSpan.FromMinutes(2) }};

	private readonly HttpClient _http;
	private readonly string _baseUrl;
	private readonly bool _checkValues;
	private readonly string _exportDir;

	public {cls}Client(string baseUrl = "http://localhost:8080", HttpClient? http = null, bool checkValues = false, string? exportDir = null)
	{{
		_http = http ?? SharedHttp;
		_baseUrl = baseUrl.TrimEnd('/');
		_checkValues = checkValues;
		_exportDir = exportDir ?? AppContext.BaseDirectory;
	}}

	public async Task<{result_type}> RunAsync({params})
	{{
		{read}var payload = new JsonObject
		{{
			["messages"] = new JsonArray(
				new JsonObject {{ ["role"] = "system", ["content"] = SystemPrompt }},
				new JsonObject {{ ["role"] = "user", ["content"] = {user} }}),
			["temperature"] = 0,
		}};
		using var content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
		using var response = await _http.PostAsync(_baseUrl + "/v1/chat/completions", content, cancellationToken);
		response.EnsureSuccessStatusCode();
		using var reply = JsonDocument.Parse(await response.Content.ReadAsStringAsync(cancellationToken));
		var text = reply.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? "null";
		if (_checkValues)
		{{
			CheckValues(text);
		}}
		{deserialize}
	}}
{csharp_checks(refs)}}}

{classes}
"""


def python_server_client(name, system_prompt, vision, refs, serve):
	fn = snake(name)
	if vision:
		imports = "import base64\nimport json\nimport mimetypes\nimport urllib.request\nfrom pathlib import Path"
		params = 'image_path, text=""'
		read = (
			'\twith open(image_path, "rb") as f:\n'
			'\t\timage = base64.b64encode(f.read()).decode("ascii")\n'
			'\tmedia = mimetypes.guess_type(image_path)[0] or "image/jpeg"\n'
			'\tuser = [{"type": "image_url", "image_url": {"url": f"data:{media};base64,{image}"}}]\n'
			'\tif text:\n'
			'\t\tuser.append({"type": "text", "text": text})\n'
		)
		user, example = "user", f'{fn}("photo.jpg")'
	else:
		imports = "import json\nimport urllib.request\nfrom pathlib import Path"
		params, read, user, example = "text", "", "text", f'{fn}("Text to analyze")'
	return f'''"""Calls the fine-tuned "{name}" model through llama.cpp's llama-server. Start it from the export folder:
	{serve}
With grammar.gbnf every data set field answers from its list. check=True re-checks answers against
datasets/, e.g. with Ollama or a server started without the grammar."""
{imports}

# {PROMPT_NOTE}
SYSTEM_PROMPT = {system_prompt!r}
{python_checks(refs)}

def {fn}({params}, host="http://localhost:8080", check=False):
{read}	payload = {{
		"messages": [{{"role": "system", "content": SYSTEM_PROMPT}}, {{"role": "user", "content": {user}}}],
		"temperature": 0,
	}}
	request = urllib.request.Request(
		host.rstrip("/") + "/v1/chat/completions",
		data=json.dumps(payload).encode("utf-8"),
		headers={{"Content-Type": "application/json"}},
	)
	with urllib.request.urlopen(request, timeout=120) as response:
		reply = json.load(response)
	result = json.loads(reply["choices"][0]["message"]["content"])
	return check_values(result) if check else result


if __name__ == "__main__":
	print({example})
'''
