"""On-disk layout: <workspace>/projects/<id>/{project.json, examples.jsonl, runs/<run id>/...}."""

import copy
import json
import os
import re
import shutil
import time
from pathlib import Path

from .data import normalize_example
from .fileio import read_json, read_jsonl, write_json, write_jsonl
from .images import remove_orphans
from .schema_tools import get_template

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")

DEFAULT_TRAINING = {
	"preset": "balanced",
	"epochs": 3,
	"learning_rate": 2e-4,
	"lora_r": 16,
	"lora_alpha": 32,
	"lora_dropout": 0.0,
	"batch_size": 4,
	"grad_accum": 2,
	"max_seq_length": 0,
	"eval_fraction": 0.1,
	"quantization": "auto",
	"accelerator": "auto",
	"seed": 42,
}

PRESETS = {
	"quick": {"epochs": 2, "learning_rate": 3e-4, "lora_r": 8, "lora_alpha": 16},
	"balanced": {"epochs": 3, "learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 32},
	"thorough": {"epochs": 5, "learning_rate": 1e-4, "lora_r": 32, "lora_alpha": 64},
}

DEFAULT_SETTINGS = {"hf_token": "", "llama_cpp_dir": ""}


def default_workspace():
	env = os.environ.get("LLMZEMPIC_HOME")
	return Path(env).expanduser() if env else Path.home() / "Llmzempic"


def slugify(name):
	slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
	return slug[:60] or "project"


def default_config(name="", template_id="classification"):
	template = get_template(template_id)
	input_kind = template.get("input_kind", "text")
	return {
		"name": name,
		"base_model": recommended_model(input_kind),
		"input_kind": input_kind,
		"instructions": template["instructions"],
		"schema": copy.deepcopy(template["schema"]),
		"include_schema_in_prompt": True,
		"training": dict(DEFAULT_TRAINING),
	}


class Project:
	def __init__(self, path):
		self.path = Path(path)
		self.id = self.path.name

	@property
	def config_path(self):
		return self.path / "project.json"

	@property
	def examples_path(self):
		return self.path / "examples.jsonl"

	@property
	def runs_dir(self):
		return self.path / "runs"

	@property
	def images_dir(self):
		return self.path / "images"

	def load_config(self):
		stored = read_json(self.config_path, {}) or {}
		config = default_config(stored.get("name") or self.id)
		config.update({k: v for k, v in stored.items() if k != "training"})
		config["training"].update(stored.get("training") or {})
		return config

	def save_config(self, config):
		current = self.load_config()
		current.update({k: v for k, v in config.items() if k != "training"})
		current["training"].update(config.get("training") or {})
		current["updated"] = time.time()
		write_json(self.config_path, current)
		return current

	def load_examples(self):
		return read_jsonl(self.examples_path)

	def save_examples(self, examples):
		rows = [normalize_example(e) for e in examples]
		write_jsonl(self.examples_path, rows)
		remove_orphans(self.images_dir, {r["image"] for r in rows if r.get("image")})
		return rows

	def run_path(self, run_id):
		if not SAFE_ID.match(run_id or ""):
			raise KeyError(run_id)
		path = self.runs_dir / run_id
		if not path.is_dir():
			raise KeyError(run_id)
		return path

	def run_ids(self):
		if not self.runs_dir.exists():
			return []
		return sorted((p.name for p in self.runs_dir.iterdir() if p.is_dir()), reverse=True)

	def new_run_dir(self):
		base = time.strftime("%Y%m%d-%H%M%S")
		run_id, n = base, 1
		while (self.runs_dir / run_id).exists():
			n += 1
			run_id = f"{base}-{n}"
		path = self.runs_dir / run_id
		path.mkdir(parents=True)
		return path


class Workspace:
	def __init__(self, root=None):
		self.root = Path(root or default_workspace()).expanduser().resolve()
		self.projects_dir = self.root / "projects"
		self.tools_dir = self.root / "tools"
		self.projects_dir.mkdir(parents=True, exist_ok=True)

	@property
	def settings_path(self):
		return self.root / "settings.json"

	def load_settings(self):
		settings = dict(DEFAULT_SETTINGS)
		settings.update(read_json(self.settings_path, {}) or {})
		return settings

	def save_settings(self, values):
		settings = self.load_settings()
		settings.update({k: v for k, v in values.items() if k in DEFAULT_SETTINGS and v is not None})
		write_json(self.settings_path, settings)
		return settings

	def projects(self):
		return [Project(p) for p in sorted(self.projects_dir.iterdir()) if (p / "project.json").exists()]

	def get(self, project_id):
		if not SAFE_ID.match(project_id or ""):
			raise KeyError(project_id)
		path = self.projects_dir / project_id
		if not (path / "project.json").exists():
			raise KeyError(project_id)
		return Project(path)

	def resolve(self, name_or_path):
		"""Find a project by id, by display name, or by a path to its folder."""
		path = Path(name_or_path).expanduser()
		if (path / "project.json").exists():
			return Project(path.resolve())
		for project in self.projects():
			if project.id == name_or_path or project.load_config().get("name") == name_or_path:
				return project
		raise KeyError(name_or_path)

	def create(self, name, template_id="classification"):
		slug = slugify(name)
		project_id, n = slug, 1
		while (self.projects_dir / project_id).exists():
			n += 1
			project_id = f"{slug}-{n}"
		path = self.projects_dir / project_id
		path.mkdir(parents=True)
		project = Project(path)
		write_json(project.config_path, {**default_config(name, template_id), "created": time.time()})
		write_jsonl(project.examples_path, [])
		return project

	def delete(self, project_id):
		project = self.get(project_id)
		shutil.rmtree(project.path)


def load_catalog():
	with open(Path(__file__).with_name("catalog.json"), "r", encoding="utf-8") as f:
		return json.load(f)["models"]


def recommended_model(input_kind="text"):
	vision = input_kind == "image"
	models = [m for m in load_catalog() if bool(m.get("vision")) == vision] or load_catalog()
	return next((m["id"] for m in models if m.get("recommended")), models[0]["id"])


def catalog_entry(model_id):
	for model in load_catalog():
		if model["id"] == model_id:
			return model
	return None
