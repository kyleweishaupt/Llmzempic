"""A project's examples in SQLite: paging, filters and search, bulk operations with undo, and
validation results cached per checks key. Replaces examples.jsonl, which is migrated once and kept
as examples.jsonl.bak."""

import json
import re
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from .data import new_id, normalize_example
from .dataset import input_key, value_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
	rowid INTEGER PRIMARY KEY,
	id TEXT UNIQUE NOT NULL,
	position REAL NOT NULL,
	image TEXT,
	input_key TEXT NOT NULL,
	input_len INTEGER NOT NULL DEFAULT 0,
	errors TEXT,
	checked TEXT,
	deleted_op TEXT,
	updated REAL,
	output TEXT NOT NULL,
	input TEXT NOT NULL -- last: long inputs spill onto overflow pages, which reading any later column must walk
);
CREATE INDEX IF NOT EXISTS examples_position ON examples(position);
CREATE INDEX IF NOT EXISTS examples_live ON examples(deleted_op, position);
CREATE INDEX IF NOT EXISTS examples_live_key ON examples(deleted_op, input_key);
CREATE INDEX IF NOT EXISTS examples_live_errors ON examples(deleted_op, errors);
CREATE INDEX IF NOT EXISTS examples_live_checked ON examples(deleted_op, checked);
CREATE INDEX IF NOT EXISTS examples_live_len ON examples(deleted_op, input_len);
CREATE VIRTUAL TABLE IF NOT EXISTS examples_fts USING fts5(input, output, content='examples', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS examples_ai AFTER INSERT ON examples BEGIN
	INSERT INTO examples_fts(rowid, input, output) VALUES (new.rowid, new.input, new.output);
END;
CREATE TRIGGER IF NOT EXISTS examples_ad AFTER DELETE ON examples BEGIN
	INSERT INTO examples_fts(examples_fts, rowid, input, output) VALUES ('delete', old.rowid, old.input, old.output);
END;
CREATE TRIGGER IF NOT EXISTS examples_au AFTER UPDATE OF input, output ON examples BEGIN
	INSERT INTO examples_fts(examples_fts, rowid, input, output) VALUES ('delete', old.rowid, old.input, old.output);
	INSERT INTO examples_fts(rowid, input, output) VALUES (new.rowid, new.input, new.output);
END;
CREATE TABLE IF NOT EXISTS ops (op TEXT PRIMARY KEY, kind TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS op_rows (op TEXT NOT NULL, id TEXT NOT NULL, prev_output TEXT);
CREATE INDEX IF NOT EXISTS op_rows_op ON op_rows(op);
"""

LIVE = "deleted_op IS NULL"
# For queries that read every live row anyway (JSON in output): unary + keeps SQLite from answering
# the term through an index, which at 200k rows means random lookups instead of one pass in rowid order.
SCAN_LIVE = "+deleted_op IS NULL"
COLUMNS = "id, input, image, output, errors"
KEEP_OPS = 20
DUPLICATE_KEYS = (
	"SELECT input_key FROM examples WHERE deleted_op IS NULL AND input_key != '' "
	"GROUP BY input_key HAVING count(*) > 1"
)


class Filter:
	"""kind: all | problems | duplicates. label: {"field", "value", "kind": enum|boolean|tags|ref}."""

	def __init__(self, kind="all", search="", label=None):
		self.kind = kind or "all"
		self.search = search or ""
		self.label = label


def _json_path(field):
	return '$."' + str(field).replace('"', '\\"') + '"'


def _json_type(value):
	"""SQLite json_type() name and comparable SQL value, so 1, true and "1" stay distinct."""
	if value is True:
		return "true", None
	if value is False:
		return "false", None
	if value is None:
		return "null", None
	if isinstance(value, int):
		return "integer", value
	if isinstance(value, float):
		return "real", value
	return "text", str(value)


def _label_clause(label):
	path = _json_path(label["field"])
	kind_name, value = _json_type(label.get("value"))
	if label.get("kind") == "ref":
		return "EXISTS (SELECT 1 FROM json_tree(output, ?) j WHERE j.type = 'text' AND j.atom = ?)", [path, value]
	if label.get("kind") == "tags":
		sql = "EXISTS (SELECT 1 FROM json_each(output, ?) j WHERE j.type = ?" + (" AND j.value = ?)" if value is not None else ")")
		return sql, [path, kind_name] + ([value] if value is not None else [])
	if value is None:
		return "json_type(output, ?) = ?", [path, kind_name]
	return "(json_type(output, ?) = ? AND json_extract(output, ?) = ?)", [path, kind_name, path, value]


def _where(flt):
	flt = flt or Filter()
	clauses, params = [SCAN_LIVE if flt.label else LIVE], []
	if flt.kind == "problems":
		clauses.append("errors IS NOT NULL AND errors != '[]'")
	elif flt.kind == "duplicates":
		clauses.append(f"input_key IN ({DUPLICATE_KEYS})")
	terms = re.findall(r"\w+", flt.search)
	if terms:
		clauses.append("rowid IN (SELECT rowid FROM examples_fts WHERE examples_fts MATCH ?)")
		params.append(" ".join(f'"{t}"*' for t in terms))
	if flt.label:
		sql, label_params = _label_clause(flt.label)
		clauses.append(sql)
		params += label_params
	order = "min(position) OVER (PARTITION BY input_key), position" if flt.kind == "duplicates" else "position"
	return " AND ".join(clauses), params, order


def _row(record):
	example_id, text, image, output, errors = record
	row = {"id": example_id, "input": text, "output": json.loads(output), "errors": json.loads(errors) if errors is not None else None}
	if image:
		row["image"] = image
	return row


class ExampleStore:
	def __init__(self, path):
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		with self.connect() as db:
			db.executescript(SCHEMA)
			db.execute("PRAGMA journal_mode=WAL")
		self._migrate()
		self._purge()

	def connect(self):
		return closing(sqlite3.connect(str(self.path), timeout=30))

	def _migrate(self):
		legacy = self.path.with_name("examples.jsonl")
		if not legacy.exists():
			return
		if not self.count():
			with open(legacy, "r", encoding="utf-8") as f:
				self.import_records((json.loads(line) for line in f if line.strip()), "append")
		legacy.replace(legacy.with_name("examples.jsonl.bak"))

	def _purge(self):
		"""Forget all but the newest undo steps; rows they soft-deleted are removed for good."""
		with self.connect() as db:
			old = [r[0] for r in db.execute("SELECT op FROM ops ORDER BY created DESC LIMIT -1 OFFSET ?", (KEEP_OPS,))]
			for op in old:
				db.execute("DELETE FROM examples WHERE deleted_op = ?", (op,))
				db.execute("DELETE FROM op_rows WHERE op = ?", (op,))
				db.execute("DELETE FROM ops WHERE op = ?", (op,))
			db.commit()

	# ---------- reading ----------

	def count(self, flt=None):
		where, params, _ = _where(flt)
		with self.connect() as db:
			return db.execute(f"SELECT count(*) FROM examples WHERE {where}", params).fetchone()[0]

	def page(self, flt, offset, limit):
		where, params, order = _where(flt)
		with self.connect() as db:
			records = db.execute(f"SELECT {COLUMNS} FROM examples WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?", params + [limit, offset])
			return [_row(r) for r in records]

	def get(self, example_id):
		with self.connect() as db:
			record = db.execute(f"SELECT {COLUMNS} FROM examples WHERE id = ? AND {LIVE}", (example_id,)).fetchone()
		if not record:
			raise KeyError(example_id)
		return _row(record)

	def iter_rows(self):
		with self.connect() as db:
			for record in db.execute(f"SELECT {COLUMNS} FROM examples WHERE {LIVE} ORDER BY position"):
				yield _row(record)

	def iter_unchecked(self, key, batch=500):
		"""Rows not yet validated for this checks key, in batches; safe to update rows between batches."""
		last = 0
		while True:
			with self.connect() as db:
				records = db.execute(
					f"SELECT rowid, {COLUMNS} FROM examples WHERE {LIVE} AND (checked IS NULL OR checked != ?) AND rowid > ? ORDER BY rowid LIMIT ?",
					(key, last, batch),
				).fetchall()
			if not records:
				return
			last = records[-1][0]
			yield [_row(r[1:]) for r in records]

	def checked_count(self, key):
		with self.connect() as db:
			return db.execute(f"SELECT count(*) FROM examples WHERE {LIVE} AND checked = ?", (key,)).fetchone()[0]

	def problem_count(self):
		return self.count(Filter("problems"))

	def duplicate_groups(self):
		with self.connect() as db:
			records = db.execute(
				f"SELECT input_key, id, output FROM examples WHERE {LIVE} AND input_key IN ({DUPLICATE_KEYS}) "
				"ORDER BY min(position) OVER (PARTITION BY input_key), position"
			).fetchall()
		groups = {}
		for key, example_id, output in records:
			groups.setdefault(key, []).append((example_id, value_key(json.loads(output))))
		return [{"ids": [i for i, _ in rows], "conflict": len({o for _, o in rows}) > 1} for rows in groups.values()]

	def referenced_images(self):
		with self.connect() as db:
			return {r[0] for r in db.execute("SELECT DISTINCT image FROM examples WHERE image IS NOT NULL")}

	def outputs(self, limit):
		with self.connect() as db:
			return [json.loads(r[0]) for r in db.execute(f"SELECT output FROM examples WHERE {LIVE} ORDER BY position LIMIT ?", (limit,))]

	# ---------- writing ----------

	@staticmethod
	def _insert(db, example, position):
		row = normalize_example(example)
		if db.execute("SELECT 1 FROM examples WHERE id = ?", (row["id"],)).fetchone():
			row["id"] = new_id()
		db.execute(
			"INSERT INTO examples (id, position, input, image, output, input_key, input_len, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
			(row["id"], position, row["input"], row.get("image"), json.dumps(row["output"], ensure_ascii=False), input_key(row), len(row["input"]), time.time()),
		)
		return row

	def add(self, examples):
		"""Insert examples above all existing ones, keeping their order."""
		examples = list(examples)
		with self.connect() as db:
			top = db.execute("SELECT min(position) FROM examples").fetchone()[0] or 0
			rows = [self._insert(db, e, top - len(examples) + i) for i, e in enumerate(examples)]
			db.commit()
		return [{**r, "errors": None} for r in rows]

	def update(self, example_id, changes):
		current = self.get(example_id)
		row = normalize_example({**current, **{k: changes[k] for k in ("input", "output", "image") if k in changes}})
		with self.connect() as db:
			db.execute(
				"UPDATE examples SET input = ?, image = ?, output = ?, input_key = ?, input_len = ?, errors = NULL, checked = NULL, updated = ? WHERE id = ?",
				(row["input"], row.get("image"), json.dumps(row["output"], ensure_ascii=False), input_key(row), len(row["input"]), time.time(), example_id),
			)
			db.commit()
		return self.get(example_id)

	def set_errors(self, pairs, key, unless_checked=False):
		"""unless_checked: the background checker's results, which skip rows checked for this key since it
		read them (a single edit is checked on the spot, and an older batch must not overwrite it)."""
		guard = " AND (checked IS NULL OR checked != ?)" if unless_checked else ""
		with self.connect() as db:
			db.executemany(
				f"UPDATE examples SET errors = ?, checked = ? WHERE id = ?{guard}",
				((json.dumps(errors, ensure_ascii=False), key, example_id, *([key] if unless_checked else [])) for example_id, errors in pairs),
			)
			db.commit()

	def import_records(self, records, mode):
		"""mode: append (add at the end), replace (drop everything first) or update (matching ids are
		overwritten, the rest added). Returns {"added", "updated"}."""
		added = updated = 0
		with self.connect() as db:
			if mode == "replace":
				for table in ("examples", "op_rows", "ops"):
					db.execute(f"DELETE FROM {table}")
			position = db.execute("SELECT max(position) FROM examples").fetchone()[0]
			position = -1 if position is None else position
			for record in records:
				if mode == "update" and record.get("id") and db.execute(f"SELECT 1 FROM examples WHERE id = ? AND {LIVE}", (str(record["id"]),)).fetchone():
					row = normalize_example(record)
					db.execute(
						"UPDATE examples SET input = ?, image = ?, output = ?, input_key = ?, input_len = ?, errors = NULL, checked = NULL, updated = ? WHERE id = ?",
						(row["input"], row.get("image"), json.dumps(row["output"], ensure_ascii=False), input_key(row), len(row["input"]), time.time(), row["id"]),
					)
					updated += 1
				else:
					position += 1
					self._insert(db, record, position)
					added += 1
			# One transaction: an import that fails part-way (replace included) leaves the examples as they were.
			db.commit()
		return {"added": added, "updated": updated}

	def _target(self, ids, flt):
		if ids is not None:
			return f"{LIVE} AND id IN (SELECT value FROM json_each(?))", [json.dumps(list(ids))]
		where, params, _ = _where(flt)
		return where, params

	def _new_op(self, db, kind):
		op = uuid.uuid4().hex[:12]
		db.execute("INSERT INTO ops VALUES (?, ?, ?)", (op, kind, time.time()))
		return op

	def delete(self, ids=None, flt=None):
		"""Soft delete, undoable with undo(op). Returns (op, number of examples)."""
		where, params = self._target(ids, flt)
		with self.connect() as db:
			op = self._new_op(db, "delete")
			n = db.execute(f"UPDATE examples SET deleted_op = ? WHERE {where}", [op] + params).rowcount
			db.commit()
		return op, n

	def set_output(self, output, ids=None, flt=None):
		where, params = self._target(ids, flt)
		text = json.dumps(output, ensure_ascii=False)
		with self.connect() as db:
			op = self._new_op(db, "set_output")
			db.execute(f"INSERT INTO op_rows SELECT ?, id, output FROM examples WHERE {where}", [op] + params)
			n = db.execute(
				"UPDATE examples SET output = ?, errors = NULL, checked = NULL, updated = ? WHERE id IN (SELECT id FROM op_rows WHERE op = ?)",
				(text, time.time(), op),
			).rowcount
			db.commit()
		return op, n

	def undo(self, op):
		with self.connect() as db:
			found = db.execute("SELECT kind FROM ops WHERE op = ?", (op,)).fetchone()
			if not found:
				return 0
			if found[0] == "delete":
				n = db.execute("UPDATE examples SET deleted_op = NULL WHERE deleted_op = ?", (op,)).rowcount
			else:
				n = db.execute(
					"UPDATE examples SET output = (SELECT prev_output FROM op_rows r WHERE r.op = ? AND r.id = examples.id), "
					"errors = NULL, checked = NULL, updated = ? WHERE id IN (SELECT id FROM op_rows WHERE op = ?)",
					(op, time.time(), op),
				).rowcount
			db.execute("DELETE FROM op_rows WHERE op = ?", (op,))
			db.execute("DELETE FROM ops WHERE op = ?", (op,))
			db.commit()
		return n
