"""Small JSON / JSONL helpers shared by the server and the worker processes."""

import json
import os
import time
from pathlib import Path


def read_json(path, default=None):
	path = Path(path)
	if not path.exists():
		return default
	for attempt in range(5):
		try:
			with open(path, "r", encoding="utf-8") as f:
				return json.load(f)
		except (json.JSONDecodeError, PermissionError):
			# Another process may be mid-replace (Windows) - retry briefly.
			if attempt == 4:
				return default
			time.sleep(0.05)
	return default


def write_json(path, data):
	"""Write JSON atomically so readers in other processes never see a partial file."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent="\t")
	for attempt in range(20):
		try:
			os.replace(tmp, path)
			return
		except PermissionError:
			if attempt == 19:
				raise
			time.sleep(0.05)


def read_jsonl(path):
	path = Path(path)
	if not path.exists():
		return []
	rows = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def write_jsonl(path, rows):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
	with open(tmp, "w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")
	os.replace(tmp, path)


def append_jsonl(path, row):
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tail_text(path, max_bytes=12000):
	path = Path(path)
	if not path.exists():
		return ""
	size = path.stat().st_size
	with open(path, "rb") as f:
		if size > max_bytes:
			f.seek(size - max_bytes)
		data = f.read()
	text = data.decode("utf-8", errors="replace")
	if size > max_bytes:
		text = text.split("\n", 1)[-1]
	return text


class StatusFile:
	"""A JSON status document that a worker updates and the server polls."""

	def __init__(self, path):
		self.path = Path(path)
		self.data = read_json(self.path, {}) or {}

	def update(self, **fields):
		self.data.update(fields)
		self.data["updated"] = time.time()
		write_json(self.path, self.data)
		return self.data
