"""Reference data sets: named lists of allowed values (with optional descriptions) that output
fields can draw from. Workspace-level library, one folder per set: values.db (SQLite + FTS5) and
meta.json. A set's content hash identifies its version; runs keep their own copy."""

import csv
import hashlib
import json
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from .fileio import read_json, write_json
from .refs import Resolver
from .workspace import SAFE_ID, slugify

VALUE_NAMES = ("value", "code", "id", "key")
DESCRIPTION_NAMES = ("description", "name", "label", "title", "text")
TXT_SPLIT = re.compile(r"\t+|\s{2,}")
BATCH = 1000


class DataSetError(Exception):
	pass


def _format(filename):
	suffix = Path(filename or "").suffix.lower()
	return {".csv": "csv", ".tsv": "tsv", ".json": "json"}.get(suffix, "txt")


def _pick(columns, names, fallback):
	lowered = {c.strip().lower(): c for c in columns}
	for name in names:
		if name in lowered:
			return lowered[name]
	return fallback


def _columns(header, value_column, description_column):
	"""Chosen or detected (value, description) columns. An empty description_column means none."""
	value = value_column or _pick(header, VALUE_NAMES, header[0] if header else None)
	rest = [c for c in header if c != value]
	if description_column is None:
		description = _pick(rest, DESCRIPTION_NAMES, rest[0] if rest else None)
	else:
		description = description_column or None
	if value not in header or (description is not None and description not in header):
		raise DataSetError(f"Column not found. The file has: {', '.join(header)}")
	return value, description


def _clean(value, description):
	value = "" if value is None else str(value).strip()
	if "\n" in value or "\r" in value:
		raise DataSetError(f"Values can't contain line breaks (exports list one value per line): {value[:60]!r}")
	description = None if description is None else str(description).strip() or None
	return value, description


def _txt_row(line):
	parts = TXT_SPLIT.split(line.strip(), maxsplit=1)
	return parts[0], parts[1] if len(parts) > 1 else None


def _json_row(item):
	if isinstance(item, dict):
		value_key = _pick(list(item), VALUE_NAMES, None)
		description_key = _pick(list(item), DESCRIPTION_NAMES, None)
		if value_key is None:
			raise DataSetError("JSON objects need a value, code, id or key field.")
		return item[value_key], item.get(description_key) if description_key else None
	return item, None


def preview(text, filename, value_column=None, description_column=None):
	"""Detected (or chosen) format, columns and the first rows, from the start of a file."""
	fmt = _format(filename)
	text = text.lstrip("﻿")
	info = {"format": fmt, "columns": None, "value_column": None, "description_column": None, "rows": []}
	rows = []
	if fmt in ("csv", "tsv"):
		reader = csv.reader(text.splitlines(), dialect="excel-tab" if fmt == "tsv" else "excel")
		header = next(reader, [])
		value, description = _columns(header, value_column, description_column) if header else (None, None)
		info.update(columns=header, value_column=value, description_column=description)
		for record in reader:
			cells = dict(zip(header, record))
			rows.append(_clean(cells.get(value), cells.get(description) if description else None))
			if len(rows) >= 20:
				break
	elif fmt == "json":
		try:
			data = json.loads(text)
		except ValueError:
			data = []  # a partial file; the full upload is checked properly
		rows = [_clean(*_json_row(item)) for item in (data if isinstance(data, list) else [])[:20]]
	else:
		for line in text.splitlines():
			if line.strip():
				rows.append(_clean(*_txt_row(line)))
			if len(rows) >= 20:
				break
	info["rows"] = [{"value": v, "description": d} for v, d in rows if v]
	return info


def iter_values(path, filename, value_column=None, description_column=None):
	"""Yield (value, description or None) from a CSV, TSV, TXT or JSON file, streaming where possible."""
	fmt = _format(filename)
	with open(path, "r", encoding="utf-8-sig", newline="") as f:
		if fmt in ("csv", "tsv"):
			reader = csv.DictReader(f, dialect="excel-tab" if fmt == "tsv" else "excel")
			if not reader.fieldnames:
				raise DataSetError("The file is empty.")
			value, description = _columns(reader.fieldnames, value_column, description_column)
			for record in reader:
				yield _clean(record.get(value), record.get(description) if description else None)
		elif fmt == "json":
			try:
				data = json.load(f)
			except ValueError as e:
				raise DataSetError(f"Not valid JSON: {e}")
			if not isinstance(data, list):
				raise DataSetError("A JSON data set must be an array of values or objects.")
			for item in data:
				yield _clean(*_json_row(item))
		else:
			for line in f:
				if line.strip():
					yield _clean(*_txt_row(line))


def _connect(path):
	return closing(sqlite3.connect(str(path), timeout=30))


def _build(folder, rows):
	"""Write values.db into folder. Returns (row count, duplicates, content hash)."""
	total = 0
	with _connect(folder / "values.db") as db:
		db.execute("CREATE TABLE rows (value TEXT UNIQUE NOT NULL, description TEXT)")
		batch = []
		for value, description in rows:
			if not value:
				continue
			total += 1
			batch.append((value, description))
			if len(batch) >= BATCH:
				db.executemany("INSERT OR IGNORE INTO rows VALUES (?, ?)", batch)
				batch = []
		db.executemany("INSERT OR IGNORE INTO rows VALUES (?, ?)", batch)
		db.execute("CREATE VIRTUAL TABLE rows_fts USING fts5(value, description, content='rows', content_rowid='rowid')")
		db.execute("INSERT INTO rows_fts(rows_fts) VALUES ('rebuild')")
		db.commit()
		count = db.execute("SELECT count(*) FROM rows").fetchone()[0]
		digest = hashlib.sha256()
		for value, description in db.execute("SELECT value, coalesce(description, '') FROM rows ORDER BY value"):
			digest.update(f"{value}\x1f{description}\n".encode("utf-8"))
	if not count:
		raise DataSetError("No values found in the file.")
	return count, total - count, digest.hexdigest()


class DataSet:
	def __init__(self, path):
		self.path = Path(path)
		self.meta = read_json(self.path / "meta.json", {}) or {}
		# A replace writes a new file beside the old one; an open file can't be moved on Windows.
		self.db_file = self.path / self.meta.get("file", "values.db")

	def _db(self):
		return _connect(self.db_file)

	def missing(self, values):
		wanted = list(dict.fromkeys(values))
		found = set()
		with self._db() as db:
			for i in range(0, len(wanted), 500):
				chunk = wanted[i:i + 500]
				marks = ",".join("?" * len(chunk))
				found.update(r[0] for r in db.execute(f"SELECT value FROM rows WHERE value IN ({marks})", chunk))
		return set(wanted) - found

	def search(self, query, limit=20):
		"""Values starting with the query first, then values whose words match it."""
		query = (query or "").strip()
		with self._db() as db:
			if not query:
				rows = db.execute("SELECT value, description FROM rows ORDER BY rowid LIMIT ?", (limit,)).fetchall()
			else:
				rows = db.execute(
					"SELECT value, description FROM rows WHERE value >= ? AND value < ? ORDER BY value LIMIT ?",
					(query, query + "￿", limit),
				).fetchall()
				terms = re.findall(r"\w+", query)
				if len(rows) < limit and terms:
					match = " ".join(f'"{t}"*' for t in terms)
					rows += db.execute(
						"SELECT value, description FROM rows_fts WHERE rows_fts MATCH ? ORDER BY rank LIMIT ?",
						(match, limit * 2),
					).fetchall()
		seen, out = set(), []
		for value, description in rows:
			if value not in seen:
				seen.add(value)
				out.append({"value": value, "description": description})
		return out[:limit]

	def iter_rows(self):
		with self._db() as db:
			yield from db.execute("SELECT value, description FROM rows ORDER BY rowid")

	def copy_to(self, dest_dir):
		dest_dir = Path(dest_dir)
		dest_dir.mkdir(parents=True, exist_ok=True)
		shutil.copy2(self.db_file, dest_dir / self.db_file.name)
		shutil.copy2(self.path / "meta.json", dest_dir / "meta.json")


def run_resolver(run_dir):
	"""A Resolver over the data set copies a run was created with."""
	folder = Path(run_dir) / "datasets"
	sets = [DataSet(p) for p in sorted(folder.iterdir()) if (p / "meta.json").is_file()] if folder.is_dir() else []
	return Resolver({d.path.name: d for d in sets})


class Library:
	def __init__(self, root):
		self.root = Path(root)

	def list(self):
		if not self.root.is_dir():
			return []
		# Dot folders are uploads being staged or sets being retired.
		return [DataSet(p).meta for p in sorted(self.root.iterdir()) if not p.name.startswith(".") and (p / "meta.json").is_file()]

	def get(self, dataset_id):
		if not SAFE_ID.match(dataset_id or ""):
			raise KeyError(dataset_id)
		path = self.root / dataset_id
		if not (path / "meta.json").is_file():
			raise KeyError(dataset_id)
		return DataSet(path)

	def resolver(self, ids=None):
		"""A Resolver over the given data set ids (all when None); ids not in the library resolve to nothing."""
		wanted = None if ids is None else set(ids)
		return Resolver({m["id"]: self.get(m["id"]) for m in self.list() if wanted is None or m["id"] in wanted})

	def by_hash(self, content_hash):
		return next((self.get(m["id"]) for m in self.list() if m.get("hash") == content_hash), None)

	def _stage(self, source_path, filename, value_column, description_column):
		self.root.mkdir(parents=True, exist_ok=True)
		staging = self.root / f".tmp-{uuid.uuid4().hex}"
		staging.mkdir()
		try:
			count, duplicates, digest = _build(staging, iter_values(source_path, filename, value_column, description_column))
		except Exception:
			shutil.rmtree(staging, ignore_errors=True)
			raise
		return staging, {"rows": count, "duplicates": duplicates, "hash": digest, "source": Path(filename).name}

	def _new_id(self, name):
		base = slugify(name)
		dataset_id, n = base, 1
		while (self.root / dataset_id).exists():
			n += 1
			dataset_id = f"{base}-{n}"
		return dataset_id

	def create(self, name, source_path, filename, value_column=None, description_column=None):
		staging, info = self._stage(source_path, filename, value_column, description_column)
		existing = self.by_hash(info["hash"])
		if existing:
			shutil.rmtree(staging, ignore_errors=True)
			return {**existing.meta, "reused": True}
		dataset_id = self._new_id(name)
		meta = {"id": dataset_id, "name": str(name).strip() or dataset_id, **info, "created": time.time()}
		write_json(staging / "meta.json", meta)
		staging.rename(self.root / dataset_id)
		return meta

	def adopt(self, folder, name):
		"""Move a ready data set folder (values.db + meta.json, e.g. from a package) in under a fresh id."""
		self.root.mkdir(parents=True, exist_ok=True)
		dataset_id = self._new_id(name)
		meta = {**(read_json(Path(folder) / "meta.json", {}) or {}), "id": dataset_id, "name": name}
		write_json(Path(folder) / "meta.json", meta)
		Path(folder).rename(self.root / dataset_id)
		return meta

	def replace(self, dataset_id, source_path, filename, value_column=None, description_column=None):
		old = self.get(dataset_id)
		staging, info = self._stage(source_path, filename, value_column, description_column)
		meta = {**old.meta, **info, "updated": time.time()}
		if info["hash"] != old.meta.get("hash"):
			meta["file"] = f"values-{info['hash'][:16]}.db"
			(staging / "values.db").replace(old.path / meta["file"])
		write_json(old.path / "meta.json", meta)
		shutil.rmtree(staging, ignore_errors=True)
		for stale in old.path.glob("values*.db"):
			if stale.name != meta.get("file", "values.db"):
				try:
					stale.unlink()
				except PermissionError:
					pass  # still open (Windows); the next replace removes it
		return meta

	def delete(self, dataset_id):
		shutil.rmtree(self.get(dataset_id).path)
