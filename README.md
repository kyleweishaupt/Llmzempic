# Llmzempic

Turn an open-source LLM into a small, specialized model that answers with **structured JSON your
applications can use for logic**: a PHP site that classifies text, a C# desktop app that analyzes
files and gets a reliable report back.

You describe the task, define the output with a JSON Schema, add examples, pick a base model and
train. Llmzempic fine-tunes with LoRA/QLoRA on your GPU, scores the result on held-out examples,
and exports a GGUF model plus ready-to-paste PHP, C# and Python clients.

Runs on **Windows and Linux** (Python 3.10+). The UI is a local web page; nothing leaves your
machine except model downloads from Hugging Face.

## Quick start

| | Windows | Linux |
|---|---|---|
| Install | `setup.cmd` | `./setup.sh` |
| Run | `start.cmd` | `./start.sh` |

Setup creates `.venv`, installs PyTorch (the CUDA build when `nvidia-smi` is found, otherwise the
CPU build) and the training libraries. The UI opens at <http://localhost:7870>. Projects are
stored in `~/Llmzempic` (`%USERPROFILE%\Llmzempic` on Windows); change it with `--workspace` or
the `LLMZEMPIC_HOME` environment variable.

Options for setup: `TORCH_INDEX_URL` picks another PyTorch build (for example ROCm on AMD
GPUs), `WITH_UNSLOTH=1` also installs [Unsloth](https://github.com/unslothai/unsloth) for faster,
leaner training on NVIDIA GPUs. Install by hand with `pip install torch` (from the right index)
then `pip install -e .[train,cuda]`.

## Workflow

1. **Task** - write instructions ("Classify the support ticket…") and the output JSON Schema.
	Start from a template (classification, tagging, field extraction, file analysis, sentiment)
	or infer the schema from your examples.
2. **Examples** - add input → expected-output pairs by hand, load an input from a file, or import:
	- JSONL: `{"input": "...", "output": {...}}` per line (also `prompt`/`completion`, or OpenAI
		`{"messages": [...]}` records)
	- JSON array of the same objects
	- CSV/TSV: an `input` (or `text`) column plus either an `output` JSON column or one column per
		output field, typed using the schema

	Every output is validated against the schema; training won't start until they all match.
	20 examples is a minimum, 100-500 covering every label and edge case is better. The tab counts
	examples per label (enum, boolean and tag fields) and flags labels with no examples and inputs
	that appear more than once, especially copies with different outputs.
3. **Model** - pick a base model. Each one shows whether it fits your GPU. Bigger is smarter but
	slower to train and run. Any Hugging Face model id or local folder works too.
4. **Train** - presets (Quick / Balanced / Thorough) or advanced settings. Live loss chart and log.
	When training ends the model is scored on held-out examples: valid JSON, matches schema, and
	exact match, with per-example predictions. Held-out examples are picked in proportion for each
	value of the first enum or boolean field, and copies of an input stay on the same side. Enum,
	boolean and tag fields also get a per-label breakdown (how often each value is recognized,
	precision) and the most common mix-ups. **Try it** runs a finished model on your own input;
	"Add as example" turns the answer into a new example to correct, so mistakes feed the next run.
5. **Export** - creates a GGUF model (Q8_0 by default), an Ollama `Modelfile`, the schema, the
	exact system prompt, a manifest, and client code. Optionally registers it with Ollama.

## Using the model from your code

The exported folder contains `integration/client.php`, `integration/<Name>Client.cs` (with typed
result classes generated from the schema) and a Python client. They call the model over HTTP:

- **Ollama** (Windows/Linux/macOS): `ollama create <name> -f Modelfile` in the export folder, then
	`POST /api/chat`. The instructions and schema are baked into the model's system prompt.
- **llama.cpp's `llama-server`**: `llama-server -m <model>.gguf`, then the OpenAI-compatible
	`POST /v1/chat/completions` with `system_prompt.txt` as the system message
	(`integration/llama-server-request.json` is a complete example). This is a single executable you
	can ship next to a desktop app.

Both clients pass the JSON Schema to the server, which constrains decoding so every response is
schema-valid JSON. Fine-tuning makes the *content* right; constrained decoding guarantees the
*shape*. Use temperature 0 for deterministic answers.

## Performance

- LoRA trains well under 1% of the weights, and the loss covers only the JSON answer, not the prompt.
- On NVIDIA GPUs: bf16/fp16, fused AdamW, gradient checkpointing, dynamic padding. "Auto"
	quantization switches to 4-bit QLoRA when the model wouldn't fit in VRAM at 16-bit.
- Unsloth is used automatically when installed.
- Training and export run in a separate process, so the UI stays responsive and GPU memory is
	freed when a job ends. One job runs at a time.
- CPU-only machines can train tiny models (SmolLM2, Qwen2.5 0.5B) for testing; use a GPU for
	1B+ models. Exported GGUF models run fine on CPUs.

## Command line

```
llmzempic                         # web UI (same as `llmzempic ui`)
llmzempic ui --port 8000 --no-browser
llmzempic train <project>         # train without the UI
llmzempic try <run folder> "text" # answer an input with a trained run
llmzempic export <run folder> --gguf q4_k_m --ollama
llmzempic doctor                  # detected GPU, memory and libraries
```

## Notes

- Gated models (Llama, Gemma) need their license accepted on Hugging Face and a token in Settings.
- GGUF conversion downloads llama.cpp's Python converter on first export (into
	`<workspace>/tools`), or uses the llama.cpp folder set in Settings. Q4_K_M/Q5_K_M need
	`llama-quantize` (from a llama.cpp release) on PATH; without it Llmzempic exports F16 and
	Ollama quantizes on import.
- Merging for export happens on the CPU in 16-bit and needs about 2 GB of RAM per billion
	parameters.

## Development

```
python -m venv .venv && .venv/bin/python -m pip install -e .[dev]
.venv/bin/python -m pytest
```

Layout: `llmzempic/server.py` (API), `web/` (UI, no build step), `trainer.py` and `exporter.py`
(run in the worker process), `runs.py` and `workspace.py` (on-disk projects),
`catalog.json` (base model list).
