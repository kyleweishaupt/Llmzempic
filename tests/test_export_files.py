import inspect
import json

from llmzempic.exporter import context_size, modelfile_text, write_deploy_files
from llmzempic.fileio import write_jsonl
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
