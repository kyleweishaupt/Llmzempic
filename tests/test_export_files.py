import importlib.util
import inspect
import io
import json
import re
import shutil
import subprocess

import pytest

from llmzempic.datasets import Library, run_resolver
from llmzempic.exporter import context_size, modelfile_text, write_deploy_files, write_value_files
from llmzempic.fileio import write_jsonl
from llmzempic.refs import prompt_schema
from llmzempic.snippets import all_snippets, csharp_client, pascal, snake

SCHEMA = {
	"type": "object",
	"properties": {
		"file_type": {"type": "string", "enum": ["log", "config"]},
		"total": {"type": ["number", "null"]},
		"issues": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {"severity": {"type": "string"}, "line": {"type": "integer"}},
				"required": ["severity"],
			},
		},
	},
	"required": ["file_type", "issues"],
}

SPEC = {
	"project_name": "File Checker",
	"base_model": "Qwen/Qwen2.5-1.5B-Instruct",
	"system_prompt": 'Check the file.\n\nSchema with """ quotes',
	"schema": SCHEMA,
}


def test_identifiers():
	assert pascal("file-checker") == "FileChecker"
	assert snake("File Checker 2") == "file_checker_2"
	assert pascal("123-model").startswith("Model")


def test_csharp_client_generates_typed_result():
	code = csharp_client("file-checker", SCHEMA)
	assert "public sealed class FileCheckerResult" in code
	assert "public List<IssuesItem> Issues { get; set; } = new();" in code
	assert "public double? Total { get; set; }" in code
	assert "public long Line { get; set; }" in code
	assert 'JsonSerializer.Deserialize<FileCheckerResult>(text)' in code
	assert code.count("{") == code.count("}")


def test_snippets_embed_schema_and_model_name():
	snippets = all_snippets("file-checker", SCHEMA, SPEC["system_prompt"], "file-checker-q8_0.gguf", 8192)
	assert "'model' => 'file-checker'" in snippets["php"]
	assert json.dumps(SCHEMA, indent=4) in snippets["php"]
	assert "llama-server -m file-checker-q8_0.gguf --port 8080 -c 8192" in snippets["llama_server"]
	namespace = {}
	exec(compile(snippets["python"].replace('if __name__ == "__main__":', "if False:"), "client.py", "exec"), namespace)
	assert namespace["SCHEMA"] == SCHEMA


def test_vision_snippets_send_the_image():
	snippets = all_snippets("hot-dog", SCHEMA, "SYS", "hot-dog-q8_0.gguf", 8192, vision=True, mmproj_file="mmproj-hot-dog-f16.gguf", image_tokens=256)
	assert "function hot_dog(string $imagePath, string $text = ''" in snippets["php"]
	assert "'images' => [base64_encode($image)]" in snippets["php"]
	assert "RunAsync(string imagePath" in snippets["csharp"] and '["images"] = new JsonArray(image)' in snippets["csharp"]
	assert snippets["csharp"].count("{") == snippets["csharp"].count("}")
	assert "--mmproj mmproj-hot-dog-f16.gguf --image-max-tokens 256" in snippets["llama_server"]
	assert '"image_url"' in snippets["llama_server"]
	namespace = {}
	exec(compile(snippets["python"].replace('if __name__ == "__main__":', "if False:"), "client.py", "exec"), namespace)
	assert list(inspect.signature(namespace["hot_dog"]).parameters) == ["image_path", "text", "host"]


def test_modelfile_escapes_triple_quotes():
	text = modelfile_text(SPEC, "model.gguf", 4096)
	assert "FROM ./model.gguf" in text
	assert text.count('"""') == 2
	assert "PARAMETER temperature 0" in text


def test_context_size_grows_with_inputs(tmp_path):
	write_jsonl(tmp_path / "train.jsonl", [{"input": "x" * 30000, "output": {}}])
	write_jsonl(tmp_path / "eval.jsonl", [])
	assert context_size(SPEC, tmp_path) == 16384


def test_context_size_leaves_room_for_images(tmp_path):
	write_jsonl(tmp_path / "train.jsonl", [{"input": "", "image": "a.jpg", "output": {}}, {"image": "b.jpg", "output": {}}])
	write_jsonl(tmp_path / "eval.jsonl", [])
	text_only = context_size(SPEC, tmp_path)
	image_spec = {**SPEC, "system_prompt": "x" * 10000, "input_kind": "image"}
	assert context_size({**image_spec, "input_kind": "text"}, tmp_path) == 4096
	assert context_size(image_spec, tmp_path) == 8192
	assert text_only == 4096


def test_vision_modelfile_adds_the_image_encoder():
	text = modelfile_text(SPEC, "model.gguf", 4096, "mmproj-model-f16.gguf")
	assert "FROM ./model.gguf\nFROM ./mmproj-model-f16.gguf\n" in text


def test_write_deploy_files_for_image_models(tmp_path):
	spec = {**SPEC, "input_kind": "image"}
	write_deploy_files(tmp_path, spec, "file-checker", "file-checker-q8_0.gguf", 4096, None, "mmproj-file-checker-f16.gguf", 256)
	manifest = json.loads((tmp_path / "llmzempic.json").read_text(encoding="utf-8"))
	assert manifest["input_kind"] == "image"
	assert manifest["mmproj"] == "mmproj-file-checker-f16.gguf" and manifest["image_max_tokens"] == 256
	assert "FROM ./mmproj-file-checker-f16.gguf" in (tmp_path / "Modelfile").read_text(encoding="utf-8")
	request = json.loads((tmp_path / "integration" / "llama-server-request.json").read_text(encoding="utf-8"))
	assert request["messages"][1]["content"][0]["type"] == "image_url"
	assert "--mmproj mmproj-file-checker-f16.gguf" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_write_deploy_files(tmp_path):
	write_deploy_files(tmp_path, SPEC, "file-checker", "file-checker-q8_0.gguf", 4096, {"json_valid": 1.0, "schema_valid": 1.0, "exact_match": 0.5, "count": 4})
	names = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
	assert {
		"schema.json", "system_prompt.txt", "Modelfile", "README.md", "llmzempic.json",
		"integration/client.php", "integration/file_checker_client.py", "integration/FileCheckerClient.cs",
		"integration/llama-server-request.json",
	} <= names
	manifest = json.loads((tmp_path / "llmzempic.json").read_text(encoding="utf-8"))
	assert manifest["schema"] == SCHEMA
	assert manifest["gguf"] == "file-checker-q8_0.gguf"
	request = json.loads((tmp_path / "integration" / "llama-server-request.json").read_text(encoding="utf-8"))
	assert request["messages"][0]["content"] == SPEC["system_prompt"]
	assert request["response_format"]["json_schema"]["schema"] == SCHEMA


CODES = {"type": "string", "x-dataset": "codes"}
REF_SCHEMA = {
	"type": "object",
	"properties": {
		"primary": {"type": "array", "items": CODES, "maxItems": 1},
		"supporting": {"type": "array", "items": CODES, "maxItems": 5},
	},
	"required": ["primary", "supporting"],
	"additionalProperties": False,
}
REF_SPEC = {**SPEC, "schema": REF_SCHEMA, "datasets": {"codes": {"name": "Codes", "hash": "h", "rows": 3}}}


def ref_export(tmp_path):
	"""Export files for a run whose schema draws on a three-value data set."""
	source = tmp_path / "codes.txt"
	source.write_text("A00\nA01\nB1\n", encoding="utf-8")
	library = Library(tmp_path / "library")
	library.create("Codes", source, "codes.txt")
	run_dir = tmp_path / "run"
	library.get("codes").copy_to(run_dir / "datasets" / "codes")
	export_dir = run_dir / "export"
	write_deploy_files(export_dir, REF_SPEC, "coder", "coder-q8_0.gguf", 4096, None)
	write_value_files(export_dir, REF_SPEC, run_resolver(run_dir))
	return export_dir


def test_export_ships_lists_and_manifest(tmp_path):
	export_dir = ref_export(tmp_path)
	assert (export_dir / "datasets" / "codes.txt").read_text(encoding="utf-8").splitlines() == ["A00", "A01", "B1"]
	manifest = json.loads((export_dir / "llmzempic.json").read_text(encoding="utf-8"))
	assert manifest["datasets"] == [
		{"path": ["primary", "*"], "file": "datasets/codes.txt", "name": "Codes"},
		{"path": ["supporting", "*"], "file": "datasets/codes.txt", "name": "Codes"},
	]
	assert manifest["grammar"] == "grammar.gbnf"
	assert manifest["schema"] == prompt_schema(REF_SCHEMA, {"codes": "Codes"})


def test_schema_json_matches_spike_path(tmp_path):
	# Decision B: the values live in a prefix-tree grammar llama-server loads once, never in the schema or a request.
	export_dir = ref_export(tmp_path)
	shipped = (export_dir / "schema.json").read_text(encoding="utf-8")
	assert json.loads(shipped) == prompt_schema(REF_SCHEMA, {"codes": "Codes"})
	assert "A01" not in shipped and "x-dataset" not in shipped
	assert '"A0" ' in (export_dir / "grammar.gbnf").read_text(encoding="utf-8")
	request = json.loads((export_dir / "integration" / "llama-server-request.json").read_text(encoding="utf-8"))
	assert "response_format" not in request
	assert "--grammar-file grammar.gbnf" in (export_dir / "README.md").read_text(encoding="utf-8")


def load_client(path):
	spec = importlib.util.spec_from_file_location(path.stem, path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def test_clients_validate_values(tmp_path, monkeypatch):
	# Spec: llama-server is the runtime that enforces the lists, so data set clients call it by default.
	export_dir = ref_export(tmp_path)
	client = load_client(export_dir / "integration" / "coder_client.py")
	answer, sent = {}, []

	def reply(request, timeout):
		sent.append((request.full_url, json.loads(request.data)))
		return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}]}).encode())

	monkeypatch.setattr(client.urllib.request, "urlopen", reply)
	answer.update(primary=["A00"], supporting=["B1", "ZZ9"])
	with pytest.raises(ValueError, match=re.escape('supporting/1: "ZZ9" isn\'t in data set Codes')):
		client.coder("Text", check=True)
	url, payload = sent[0]
	assert url == "http://localhost:8080/v1/chat/completions" and "response_format" not in payload
	assert payload["messages"] == [{"role": "system", "content": REF_SPEC["system_prompt"]}, {"role": "user", "content": "Text"}]
	assert client.coder("Text") == answer  # the grammar already enforces the lists; checking is opt-in
	answer.update(supporting=["B1", "A01"])
	assert client.coder("Text", check=True) == {"primary": ["A00"], "supporting": ["B1", "A01"]}


@pytest.mark.skipif(not shutil.which("php"), reason="php is not installed")
def test_php_client_checks_values(tmp_path):
	export_dir = ref_export(tmp_path)
	client = export_dir / "integration" / "client.php"
	result = subprocess.run(["php", "-l", str(client)], capture_output=True, text=True, timeout=60)
	assert result.returncode == 0, result.stdout + result.stderr
	script = tmp_path / "check.php"
	script.write_text(
		f"<?php require {json.dumps(str(client))};\n"
		"coder_check_values(['primary' => ['A00'], 'supporting' => ['B1']]);\n"
		"try { coder_check_values(['primary' => ['A00'], 'supporting' => ['B1', 'ZZ9']]); }\n"
		"catch (UnexpectedValueException $e) { echo $e->getMessage(); }\n",
		encoding="utf-8",
	)
	result = subprocess.run(["php", str(script)], capture_output=True, text=True, timeout=60)
	assert result.stdout == 'supporting/1: "ZZ9" isn\'t in data set Codes', result.stdout + result.stderr


@pytest.mark.skipif(not shutil.which("dotnet"), reason="dotnet is not installed")
def test_csharp_client_with_values_builds(tmp_path):
	export_dir = ref_export(tmp_path)
	sdks = subprocess.run(["dotnet", "--list-sdks"], capture_output=True, text=True, timeout=60).stdout.split()
	major = max(int(v.split(".")[0]) for v in sdks if re.match(r"\d+\.\d+\.\d+$", v))
	project = tmp_path / "cs"
	project.mkdir()
	shutil.copy(export_dir / "integration" / "CoderClient.cs", project)
	(project / "Check.csproj").write_text(
		f'<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net{major}.0</TargetFramework>'
		"<Nullable>enable</Nullable><TreatWarningsAsErrors>true</TreatWarningsAsErrors></PropertyGroup></Project>",
		encoding="utf-8",
	)
	result = subprocess.run(["dotnet", "build", str(project), "-v", "q", "--nologo"], capture_output=True, text=True, timeout=300)
	assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
