# Llmzempic

Turn an open-source LLM into a small, specialized model that answers with **structured JSON your
applications can use for logic**: a PHP site that classifies text, a C# desktop app that analyzes
files and gets a reliable report back, or an image check ("does this photo contain a hot dog?").

You describe the task, design the output in a visual editor (it's a JSON Schema underneath), add
examples (text or images), pick a base model and train. Llmzempic fine-tunes with LoRA/QLoRA on your GPU, scores the result on held-out examples,
and exports a GGUF model plus ready-to-paste PHP, C# and Python clients.

Runs on **Windows and Linux** (Python 3.10+). The UI is a local web page; nothing leaves your
machine except model downloads from Hugging Face.

## Quick start

| | Windows | Linux |
|---|---|---|
| Install | `setup.cmd` | `./setup.sh` |
| Run | `start.cmd` | `./start.sh` |

Setup creates `.venv`, installs PyTorch and torchvision (the CUDA build when `nvidia-smi` is
found, otherwise the CPU build) and the training libraries. The UI opens at <http://localhost:7870>. Projects are
stored in `~/Llmzempic` (`%USERPROFILE%\Llmzempic` on Windows); change it with `--workspace` or
the `LLMZEMPIC_HOME` environment variable.

Options for setup: `TORCH_INDEX_URL` picks another PyTorch build (for example ROCm on AMD
GPUs), `WITH_UNSLOTH=1` also installs [Unsloth](https://github.com/unslothai/unsloth) for faster,
leaner training on NVIDIA GPUs. Install by hand with `pip install torch` (from the right index)
then `pip install -e .[train,cuda]`.

## Workflow

1. **Task** - choose the input (text, or images with optional text) and the answer (JSON, or
	[decisions](#decision-models) with calibrated probabilities), write instructions
	("Classify the support ticket…") and design the answer. The visual editor covers text, numbers,
	yes/no, one or several of a list of options, values from a [data set](#data-sets), lists (with
	at least / at most counts) and nested groups, with a live preview of the answer; switch to the raw
	JSON Schema for anything else. Start from a template (classification,
	tagging, field extraction, file analysis, sentiment, image yes/no, image classification) or infer
	the schema from your examples.
2. **Examples** - each answer is edited with a form generated from the schema (or as JSON).
	Text examples are added by hand, loaded from a file, or imported:
	- JSONL: `{"input": "...", "output": {...}}` per line (also `prompt`/`completion`, or OpenAI
		`{"messages": [...]}` records)
	- JSON array of the same objects
	- CSV/TSV: an `input` (or `text`) column plus either an `output` JSON column or one column per
		output field, typed using the schema

	Image projects: drop, paste or choose a batch of images and give the whole batch its answer
	(add the hot dogs as "yes", then the rest as "no"). Big photos are rotated per their EXIF data
	and shrunk to 1280 px in the browser before upload; identical images are stored once. The gallery
	has answer buttons on every card, Shift-click range selection to set an answer for or delete many
	at once (with undo), and a full-size review view for fast labeling: `1`-`9` set the first yes/no
	or option field and move on, `←`/`→` browse, `Del` deletes.

	Every output is validated against the schema (and its data sets) in the background; training
	won't start until the check finishes and they all match.
	20 examples is a minimum, 100-500 covering every label and edge case is better. The tab counts
	examples per label (enum, boolean and tag fields) and flags labels with no examples and inputs
	that appear more than once, especially copies with different outputs.
3. **Model** - pick a base model: text models for text projects, vision models (SmolVLM, Qwen3-VL)
	for image projects. Each one shows whether it fits your GPU. Bigger is smarter but
	slower to train and run. Any Hugging Face model id or local folder works too.
4. **Train** - presets (Quick / Balanced / Thorough) or advanced settings. Live loss chart and log.
	When training ends the model is scored on held-out examples: valid JSON, matches schema, and
	exact match, with per-example predictions. Held-out examples are picked in proportion for each
	value of the first enum or boolean field, and copies of an input stay on the same side. Enum,
	boolean and tag fields also get a per-label breakdown (how often each value is recognized,
	precision) and the most common mix-ups; data set fields get exact, first value, precision, recall,
	F1 and not-in-the-set rates, plus a per-value table that opens the matching examples. **Try it** runs a finished model on your own input or
	image; "Add as example" turns the answer into a new example to correct, so mistakes feed the next
	run.
5. **Export** - creates a GGUF model (Q8_0 by default), an Ollama `Modelfile`, the schema, the
	exact system prompt, a manifest, and client code. Image models also get their image encoder as
	`mmproj-<name>-f16.gguf`. Optionally registers the model with Ollama.

## Decision models

Many apps don't need generated text, just a fast typed decision: route a ticket, rate its urgency, flag
a policy violation. Choose **Decisions** as the answer on the Task tab and every top-level output field
becomes a question, phrased by the field's description:

- an option list is a **choice** (up to 255 options), or an **ordered score** (a rubric of 2-10 levels;
	its score is the expected level, for example 2.7 on a 1-4 scale);
- a yes/no field is a **predicate**: the probability that it's true.

Each question is asked on its own: the prompt numbers the options and the model answers with one number.
The probability of every option is read from one batched forward pass, so nothing is generated and a
decision costs input tokens only. Numbers scale to hundreds of options (zero-padded, scored over all of
their tokens), where letters run out.

- **Training** makes one sequence per example and question, reshuffling choice options each time so the
	model reads the list instead of memorizing positions. A KL anchor (*KL anchor weight* on the Model tab,
	0.1 by default, 0 turns it off) keeps the model's next-token predictions on the prompt close to the base
	model's, so learning to decide costs little general ability. Unsloth isn't used for decision models.
- **Scoring** reports accuracy per question, calibration (expected calibration error, Brier score) and
	latency, next to the base model reading the same prompts without fine-tuning. With 50 or more held-out
	answers, a temperature is fitted to calibrate the probabilities (cross-fitted for the scores shown) and
	applied whenever the run answers. **Try it** shows every option's probability.
- **Serving**: `llmzempic serve --run <run folder>` keeps the model loaded and answers
	`POST http://localhost:7871/v1/decisions`; `--model <base model>` serves any base model training-free.
	Requests can leave `questions` out (all trained questions), name a trained question only, or bring new
	questions (answered, but not calibrated). `min_confidence` withholds less confident answers
	(`"refused": true`, probabilities still reported). The input can be text, JSON, or chat messages.

```
{"input": "I was charged twice, refund me!", "min_confidence": 0.6,
 "questions": [{"name": "intent", "type": "choice", "instructions": "What does the customer want?",
                "choices": ["billing", "technical_support", "sales", "other"]},
               {"name": "urgency", "type": "score", "instructions": "Rate urgency", "levels": ["low", "medium", "high"]},
               {"name": "is_toxic", "type": "predicate", "instructions": "Is the message toxic or offensive?"}]}

{"answers": [{"name": "intent", "type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, ...}, "confidence": 0.91},
             {"name": "urgency", "type": "score", "score": 2.4, "level": "medium", "probabilities": {...}, "confidence": 0.55},
             {"name": "is_toxic", "type": "predicate", "probability": 0.03, "confidence": 0.97}],
 "usage": {"input_tokens": 412, "output_tokens": 0}}
```

Exporting a decision run writes `decisions.json` (questions, prompts, calibration temperature), Python,
PHP and C# clients for the Decisions API, and the GGUF model, which answers each question's prompt with an
option number. Decision models read text for now; image projects use JSON answers.

## Data sets

Some answers have to come from a fixed list: product codes, category codes, account numbers. Add the
list once on the **Data sets** page (CSV, TSV, TXT with one value per line, or JSON; for CSV pick the
value column and an optional description column) and use it in any project with the "Value from a data
set" field type, alone or as the items of a list (ordered, with at least / at most counts). Lists of a
million values are fine.

- Answers are checked against the set, and example forms pick values with a search box that matches
  values by prefix and descriptions by words.
- The model's prompt names the set ("One value from the data set …") but never lists it.
- A run keeps its own copy of every set it used, so replacing a set later doesn't change old runs.
  A set used by a project can't be deleted, and values can't contain line breaks.

## Large example sets

Examples live in SQLite (`examples.db` in the project folder), so the Examples tab pages, searches and
filters on the server: with 200,000 examples of about 5 KB each a page, a search or a value filter answers
in well under a second. Imports stream files of any size with progress (add, update by id, or replace
all), and checking runs in the background with live progress. "Select all N matching" applies a bulk
answer or delete to everything a search or filter matches, with undo.

Training streams examples from disk too. Examples longer than *Max tokens per example* are skipped
and listed in the run view (`skipped.json` in the run folder). *Use at most N examples* trains on a
sample; values with few examples are never sampled away or held out, so the model sees every value
it has examples for. At most *Held-out examples to score* (300 by default, up to 2,000) are scored.
`python tools/make_scale_fixture.py --examples 200000 --values 70000 --out <folder>` writes a synthetic
project of that size for testing.

## Resuming training

Training saves a checkpoint every tenth of the run (every 50 steps on short runs). If it stops or
fails, **Resume** on the run picks up from the last checkpoint.

## Project packages

The project menu's **Export package…** saves a `.llmz` file with the project's settings, examples,
images and data sets, optionally with the latest trained run. **Import package…** (the + next to
Projects) recreates it on another machine, for example to train a bigger model on a stronger GPU; data
sets already present with the same content are reused. **Duplicate project** makes a copy to try a
different schema or model without touching the original.

## Using the model from your code

The exported folder contains `integration/client.php`, `integration/<Name>Client.cs` (with typed
result classes generated from the schema) and a Python client. They call the model over HTTP:

- **Ollama** (Windows/Linux/macOS): `ollama create <name> -f Modelfile` in the export folder, then
	`POST /api/chat`. The instructions and schema are baked into the model's system prompt.
- **llama.cpp's `llama-server`**: `llama-server -m <model>.gguf`, then the OpenAI-compatible
	`POST /v1/chat/completions` with `system_prompt.txt` as the system message
	(`integration/llama-server-request.json` is a complete example). This is a single executable you
	can ship next to a desktop app.

For image models the clients take an image path (plus optional text) and send the image base64
encoded (Ollama `images`, llama-server `image_url`); run llama-server with `--mmproj` as shown in the
export's README.

The clients pass the JSON Schema to the server, which constrains decoding so every response is
schema-valid JSON. Fine-tuning makes the *content* right; constrained decoding guarantees the
*shape*. Use temperature 0 for deterministic answers.

Projects with data sets export each set as `datasets/<id>.txt` and a `grammar.gbnf` that spells the
allowed values as a prefix tree. Start llama-server with `--grammar-file grammar.gbnf` (a 2024 or later
build) and every data set field can only answer with values from its list; requests then carry no
`response_format`. The generated clients call llama-server. Ollama can't enforce lists that long; with
it, turn on the clients' value check, which compares every answer with `datasets/` and raises an error
naming any value that isn't on its list.

## Performance

- LoRA trains well under 1% of the weights, and the loss covers only the JSON answer, not the prompt.
- On NVIDIA GPUs: bf16/fp16, fused AdamW, gradient checkpointing, dynamic padding. "Auto"
	quantization switches to 4-bit QLoRA when the model wouldn't fit in VRAM at 16-bit.
- Unsloth is used automatically when installed (text models).
- Images are capped at about 256 tokens each. LoRA trains the language layers; the vision encoder
	stays frozen and isn't quantized in 4-bit mode.
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
llmzempic try <run folder> --image photo.jpg   # image runs; repeat --image for more
llmzempic decide --run <run folder> "text"      # decision runs: probabilities as JSON
llmzempic decide --model <base model> --questions questions.json "text"   # training-free
llmzempic serve --run <run folder> --port 7871  # Decisions API (or --model <base model>)
llmzempic export <run folder> --gguf q4_k_m --ollama
llmzempic package <project> --include-run      # save a .llmz package
llmzempic import-package <file.llmz>
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

Layout: `llmzempic/server.py` plus `*_api.py` routers (API), `web/` (UI: `index.html`, `css/` and
`js/` ES modules, no build step), `trainer.py`, `vision.py` and `exporter.py` (run in the worker
process), `decisions.py` (decision questions, prompts, answers and calibration; no PyTorch),
`decision_engine.py` (option readout, decision training and scoring) and `decision_server.py` (Decisions API), `runs.py`, `workspace.py`, `example_store.py`, `checks.py` and `images.py` (on-disk
projects), `datasets.py` and `refs.py` (data sets), `grammar.py` and `client_checks.py` (exports),
`packages.py`, `catalog.json` (base model list).
