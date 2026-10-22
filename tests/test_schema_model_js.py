import json
import shutil
import subprocess
from pathlib import Path

import pytest

from llmzempic.schema_tools import TEMPLATES

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_schema_editor_model_round_trips(tmp_path):
	templates = tmp_path / "templates.json"
	templates.write_text(json.dumps([t["schema"] for t in TEMPLATES]), encoding="utf-8")
	result = subprocess.run(
		["node", str(ROOT / "tests" / "js" / "schema_model_check.mjs"), str(ROOT / "llmzempic" / "web" / "js" / "schema-model.js"), str(templates)],
		capture_output=True, text=True,
	)
	assert result.returncode == 0, result.stdout + result.stderr
