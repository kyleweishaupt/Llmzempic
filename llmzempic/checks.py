"""Example validation (in a background thread per project, cached per checks key) and dataset
statistics computed in SQL, so neither ever loads a whole 200k-example project into memory."""

import hashlib
import json
import threading
import traceback

from .dataset import label_fields, value_key
from .example_store import LIVE, SCAN_LIVE, _json_path
from .images import image_path
from .refs import ref_fields, ref_values, referenced_ids
from .schema_tools import check_schema, validate_output

BATCH = 500
TOP_VALUES = 10
CHARS_PER_TOKEN = 3.5


def checks_key(config, hashes):
	"""Changes whenever what makes an example valid changes: input kind, schema, data set versions."""
	material = json.dumps(
		[config.get("input_kind", "text"), config.get("schema"), sorted(f"{k}:{v}" for k, v in hashes.items())],
		sort_keys=True, ensure_ascii=False,
	)
	return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _input_errors(example, input_kind, images_dir):
	if input_kind != "image":
		return [] if str(example.get("input", "")).strip() else ["Input is empty"]
	if not example.get("image"):
		return ["No image"]
	try:
		image_path(images_dir, example["image"])
	except KeyError:
		return ["Image file is missing"]
	return []


def example_errors(example, schema, input_kind, images_dir, resolver):
	if check_schema(schema):
		return []  # the schema's own problems are reported on the Task tab
	return _input_errors(example, input_kind, images_dir) + validate_output(example.get("output"), schema, resolver)


class Checker:
	"""Keeps every project's cached example errors in step with its current checks key."""

	def __init__(self):
		self._lock = threading.Lock()
		self._states = {}

	@staticmethod
	def _context(project, library):
		config = project.load_config()
		resolver = library.resolver(referenced_ids(config.get("schema")))
		return config, resolver, checks_key(config, resolver.hashes())

	def ensure(self, project, library, force=False):
		"""Start checking whatever isn't checked for the current key. Cheap when nothing changed."""
		config, resolver, key = self._context(project, library)
		with self._lock:
			state = self._states.get(project.id)
			if state and state["key"] == key and not force and (state["thread"].is_alive() or state["error"] or not self._unchecked(state)):
				return key
			if state:
				with state["write"]:
					state["stop"] = True  # an older thread can't write after this point
			state = {"key": key, "stop": False, "error": None, "write": threading.Lock(), "store": project.examples()}
			state["thread"] = threading.Thread(target=self._run, args=(project, config, resolver, state), daemon=True)
			self._states[project.id] = state
			state["thread"].start()
		return key

	@staticmethod
	def _unchecked(state):
		return state["store"].checked_count(state["key"]) < state["store"].count()

	@staticmethod
	def _run(project, config, resolver, state):
		store, key = state["store"], state["key"]
		try:
			for batch in store.iter_unchecked(key, BATCH):
				pairs = [(row["id"], example_errors(row, config["schema"], config.get("input_kind", "text"), project.images_dir, resolver)) for row in batch]
				with state["write"]:
					if state["stop"]:
						return
					store.set_errors(pairs, key, unless_checked=True)
		except Exception as e:  # the thread's boundary: report the failure instead of dying unseen
			traceback.print_exc()
			state["error"] = str(e) or type(e).__name__

	def progress(self, project_id):
		state = self._states.get(project_id)
		if not state:
			return {"running": False, "done": 0, "total": 0, "error": None}
		store = state["store"]
		done, total = store.checked_count(state["key"]), store.count()
		# A thread that has nothing left to check isn't checking, even before it exits.
		return {"running": state["thread"].is_alive() and done < total, "done": done, "total": total, "error": state["error"]}

	def wait(self, project_id, timeout=None):
		state = self._states.get(project_id)
		if state:
			state["thread"].join(timeout)

	def check_now(self, project, library, ids):
		"""Check a few examples synchronously (single edits). Returns {id: errors}."""
		config, resolver, key = self._context(project, library)
		store = project.examples()
		results = {}
		for example_id in ids:
			row = store.get(example_id)
			results[example_id] = example_errors(row, config["schema"], config.get("input_kind", "text"), project.images_dir, resolver)
		store.set_errors(results.items(), key)
		return results


def _from_sql(type_name, value):
	return {"true": True, "false": False, "null": None}.get(type_name, value)


def _label_counts(db, name, kind, values):
	path = _json_path(name)
	if kind == "tags":
		rows = db.execute(
			f"SELECT j.type, j.value, count(DISTINCT examples.rowid) FROM examples, json_each(examples.output, ?) j "
			f"WHERE {SCAN_LIVE} AND json_type(examples.output, ?) = 'array' GROUP BY 1, 2",
			(path, path),
		)
	else:
		rows = db.execute(f"SELECT json_type(output, ?), json_extract(output, ?), count(*) FROM examples WHERE {SCAN_LIVE} GROUP BY 1, 2", (path, path))
	counts = {}
	for type_name, value, n in rows:
		if type_name and type_name not in ("array", "object"):
			key = value_key(_from_sql(type_name, value))
			counts[key] = counts.get(key, 0) + n
	return {"field": name, "kind": kind, "values": [{"value": v, "count": counts.get(value_key(v), 0)} for v in values]}


def _ref_source(field):
	"""SQL yielding one row per value used in a top-level data set field, or None for deeper fields."""
	path = _json_path(field["path"][0])
	if field["path"] == (field["field"], "*"):
		return (f"SELECT j.value AS v FROM examples, json_each(examples.output, ?) j WHERE {SCAN_LIVE} "
			"AND json_type(examples.output, ?) = 'array' AND j.type = 'text'"), [path, path]
	if field["path"] == (field["field"],):
		return f"SELECT json_extract(output, ?) AS v FROM examples WHERE {SCAN_LIVE} AND json_type(output, ?) = 'text'", [path, path]
	return None


def _ref_counts(store, db, field, schema, library):
	try:
		dataset = library.get(field["dataset"])
	except KeyError:
		dataset = None
	source = _ref_source(field)
	if source:
		sql, params = source
		# One pass over the examples; totals, top values and misses all come from the per-value counts.
		db.execute(f"CREATE TEMP TABLE used AS SELECT v, count(*) AS n FROM ({sql}) GROUP BY v", params)
		total, distinct = db.execute("SELECT coalesce(sum(n), 0), count(*) FROM used").fetchone()
		top = db.execute("SELECT v, n FROM used ORDER BY n DESC, v LIMIT ?", (TOP_VALUES,)).fetchall()
		missing = distinct
		if dataset:
			db.execute("ATTACH DATABASE ? AS ds", (str(dataset.db_file),))
			missing = db.execute("SELECT count(*) FROM used WHERE v NOT IN (SELECT value FROM ds.rows)").fetchone()[0]
			db.execute("DETACH DATABASE ds")
		db.execute("DROP TABLE temp.used")
	else:
		# Fields nested inside lists of objects are rare; count them by streaming the outputs.
		used = {}
		for row in store.iter_rows():
			for location, dataset_id, value in ref_values(row["output"], schema):
				if location.split("/")[0] == field["field"] and dataset_id == field["dataset"]:
					used[value] = used.get(value, 0) + 1
		total, distinct = sum(used.values()), len(used)
		top = sorted(used.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_VALUES]
		missing = len(dataset.missing(used)) if dataset else distinct
	return {
		"field": field["field"], "dataset": field["dataset"],
		"name": (dataset.meta.get("name") if dataset else None) or field["dataset"],
		"distinct": distinct, "size": dataset.meta.get("rows", 0) if dataset else 0,
		"total": total, "missing": missing, "top": [{"value": v, "count": n} for v, n in top],
	}


def _input_lengths(db, max_seq_length):
	n = db.execute(f"SELECT count(*) FROM examples WHERE {LIVE}").fetchone()[0]

	def at(fraction):
		if not n:
			return 0
		offset = min(n - 1, int(n * fraction))
		return db.execute(f"SELECT input_len FROM examples WHERE {LIVE} ORDER BY input_len LIMIT 1 OFFSET ?", (offset,)).fetchone()[0]

	long_inputs = None
	if max_seq_length:
		limit = int(max_seq_length * CHARS_PER_TOKEN)
		long_inputs = db.execute(f"SELECT count(*) FROM examples WHERE {LIVE} AND input_len > ?", (limit,)).fetchone()[0]
	return {"long_inputs": long_inputs, "input_chars": {"p50": at(0.5), "p95": at(0.95), "max": at(1.0)}}


def stats(project, library, config):
	store = project.examples()
	schema = config.get("schema")
	usable = not check_schema(schema)
	groups = store.duplicate_groups()
	max_seq_length = int((config.get("training") or {}).get("max_seq_length") or 0)
	with store.connect() as db:
		fields = [_label_counts(db, *f) for f in label_fields(schema)] if usable else []
		refs = [_ref_counts(store, db, f, schema, library) for f in ref_fields(schema)] if usable else []
		lengths = _input_lengths(db, max_seq_length)
	return {
		"count": store.count(),
		"problems": store.problem_count(),
		"fields": fields,
		"refs": refs,
		"duplicates": {"groups": len(groups), "conflicts": sum(1 for g in groups if g["conflict"])},
		**lengths,
	}
