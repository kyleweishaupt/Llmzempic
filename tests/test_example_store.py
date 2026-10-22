import json

import pytest

from llmzempic.example_store import COLUMNS, LIVE, ExampleStore, Filter, _where


@pytest.fixture
def store(tmp_path):
	return ExampleStore(tmp_path / "examples.db")


def ex(i, output, **extra):
	return {"id": f"e{i}", "input": f"text {i}", "output": output, **extra}


def ids(rows):
	return [r["id"] for r in rows]


def test_migrates_jsonl_in_order_and_keeps_backup(tmp_path):
	lines = [
		{"id": "a", "input": "one", "output": {"x": 1}},
		{"input": "two", "output": {"x": 2}},
		{"id": "a", "input": "three", "output": {"x": 3}},
		{"id": "img", "input": "", "image": "0123456789abcdef01234567.jpg", "output": {"x": 4}},
	]
	(tmp_path / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
	rows = list(ExampleStore(tmp_path / "examples.db").iter_rows())
	assert [r["input"] for r in rows] == ["one", "two", "three", ""]
	assert len(set(ids(rows))) == 4 and rows[0]["id"] == "a" and rows[3]["id"] == "img"
	assert rows[3]["image"] == "0123456789abcdef01234567.jpg" and rows[2]["output"] == {"x": 3}
	assert (tmp_path / "examples.jsonl.bak").exists() and not (tmp_path / "examples.jsonl").exists()
	assert ExampleStore(tmp_path / "examples.db").count() == 4


def test_add_puts_new_examples_first(store):
	store.add([ex(1, {})])
	store.add([ex(2, {}), ex(3, {})])
	assert ids(store.page(Filter(), 0, 10)) == ["e2", "e3", "e1"]
	assert ids(store.page(Filter(), 1, 1)) == ["e3"]


def test_filters(store):
	store.import_records([
		ex(1, {"label": 1}), ex(2, {"label": True}), ex(3, {"label": "1"}),
		ex(4, {"tags": ["x", "y"]}), ex(5, {"a": ["V1"], "b": ["V2", "V3"]}),
	], "append")
	label = lambda value, kind="enum", field="label": Filter(label={"field": field, "value": value, "kind": kind})
	assert ids(store.page(label(1), 0, 10)) == ["e1"]
	assert ids(store.page(label(True, "boolean"), 0, 10)) == ["e2"]
	assert ids(store.page(label("1"), 0, 10)) == ["e3"]
	assert ids(store.page(label("y", "tags", "tags"), 0, 10)) == ["e4"]
	assert ids(store.page(label("V3", "ref", "b"), 0, 10)) == ["e5"]
	assert store.count(label("V3", "ref", "a")) == 0
	assert ids(store.page(Filter(search="text 3"), 0, 10)) == ["e3"]
	assert ids(store.page(Filter(search="V2"), 0, 10)) == ["e5"]
	store.set_errors([("e1", ["bad"]), ("e2", [])], "k1")
	assert ids(store.page(Filter("problems"), 0, 10)) == ["e1"]
	assert store.problem_count() == 1


def test_duplicates_listed_together_with_conflict_flag(store):
	store.import_records([
		{"id": "a", "input": "same", "output": {"x": 1}},
		{"id": "b", "input": "other", "output": {"x": 1}},
		{"id": "c", "input": "dup", "output": {"x": 1}},
		{"id": "d", "input": " same ", "output": {"x": 1}},
		{"id": "e", "input": "dup", "output": {"x": 2}},
	], "append")
	assert store.duplicate_groups() == [{"ids": ["a", "d"], "conflict": False}, {"ids": ["c", "e"], "conflict": True}]
	assert ids(store.page(Filter("duplicates"), 0, 10)) == ["a", "d", "c", "e"]
	assert store.count(Filter("duplicates")) == 4


def test_delete_and_set_output_undo(store):
	store.import_records([ex(i, {"x": i}) for i in range(1, 6)], "append")
	op, n = store.delete(ids=["e1", "e2"])
	assert n == 2 and store.count() == 3
	assert store.undo(op) == 2 and ids(store.page(Filter(), 0, 10)) == ["e1", "e2", "e3", "e4", "e5"]
	op, n = store.delete(flt=Filter(search="text"))
	assert n == 5 and store.count() == 0
	assert store.undo(op) == 5 and store.count() == 5
	op, n = store.set_output({"x": 9}, ids=["e3"])
	assert n == 1 and store.get("e3")["output"] == {"x": 9}
	store.undo(op)
	assert store.get("e3")["output"] == {"x": 3}
	op, n = store.set_output({"x": 0}, flt=Filter())
	assert n == 5
	store.undo(op)
	assert [r["output"]["x"] for r in store.iter_rows()] == [1, 2, 3, 4, 5]


def test_import_modes(store):
	assert store.import_records([ex(1, {"x": 1}), ex(2, {"x": 2})], "append") == {"added": 2, "updated": 0}
	result = store.import_records([ex(2, {"x": 20}), ex(3, {"x": 3}), ex(1, {"x": 10})], "update")
	assert result == {"added": 1, "updated": 2}
	assert store.get("e1")["output"] == {"x": 10} and store.count() == 3
	store.import_records([ex(1, {"x": 100})], "append")
	assert store.count() == 4 and store.get("e1")["output"] == {"x": 10}
	assert store.import_records([ex(9, {})], "replace") == {"added": 1, "updated": 0}
	assert ids(store.iter_rows()) == ["e9"]


def test_update_resets_errors(store):
	store.add([ex(1, {"x": 1})])
	store.set_errors([("e1", ["bad"])], "k")
	row = store.update("e1", {"output": {"x": 2}})
	assert row["errors"] is None and row["output"] == {"x": 2}
	with pytest.raises(KeyError):
		store.update("nope", {"input": "x"})


def test_iter_unchecked_by_key(store):
	store.import_records([ex(i, {}) for i in range(1, 4)], "append")
	store.set_errors([("e1", [])], "k1")
	assert [ids(batch) for batch in store.iter_unchecked("k1", batch=1)] == [["e2"], ["e3"]]
	assert sum(len(b) for b in store.iter_unchecked("k2")) == 3


def test_referenced_images_and_outputs(store):
	store.import_records([ex(1, {"x": 1}, image="a0a0a0a0a0a0a0a0a0a0a0a0.jpg"), ex(2, {"x": 2})], "append")
	op, _ = store.delete(ids=["e1"])
	assert store.referenced_images() == {"a0a0a0a0a0a0a0a0a0a0a0a0.jpg"}  # undo may still bring it back
	assert store.outputs(10) == [{"x": 2}]


def test_listing_queries_use_indexes_not_rows(store):
	# At 200k examples of ~5 KB, touching table rows for a count or a sort costs seconds per request.
	store.import_records([ex(i, {"x": i}) for i in range(1, 4)], "append")
	where, params, order = _where(Filter())
	counts = {
		"all": (f"SELECT count(*) FROM examples WHERE {where}", params),
		"problems": (f"SELECT count(*) FROM examples WHERE {_where(Filter('problems'))[0]}", []),
		"duplicates": (f"SELECT count(*) FROM examples WHERE {_where(Filter('duplicates'))[0]}", []),
		"checked": (f"SELECT count(*) FROM examples WHERE {LIVE} AND checked = ?", ["k"]),
		"lengths": (f"SELECT input_len FROM examples WHERE {LIVE} ORDER BY input_len LIMIT 1 OFFSET 1", []),
	}
	with store.connect() as db:
		def plan(sql, args):
			return " | ".join(r[3] for r in db.execute("EXPLAIN QUERY PLAN " + sql, args))

		assert "TEMP B-TREE" not in plan(f"SELECT {COLUMNS} FROM examples WHERE {where} ORDER BY {order} LIMIT 50 OFFSET 0", params)
		for name, (sql, args) in counts.items():
			steps = plan(sql, args)
			assert "COVERING INDEX" in steps and "TEMP B-TREE" not in steps, f"{name}: {steps}"


def test_label_filter_count_scans_rows_in_order(store):
	# Every row's output has to be read anyway; through an index that is 200k random lookups.
	store.import_records([ex(i, {"x": i}) for i in range(1, 4)], "append")
	where, params, _ = _where(Filter(label={"field": "x", "value": 1, "kind": "enum"}))
	with store.connect() as db:
		steps = " | ".join(r[3] for r in db.execute(f"EXPLAIN QUERY PLAN SELECT count(*) FROM examples WHERE {where}", params))
	assert steps.startswith("SCAN examples"), steps


def test_import_is_all_or_nothing(store):
	# A replace that fails part-way must leave the original examples, not a partial new set.
	store.import_records([ex(i, {"x": i}) for i in range(300)], "append")

	def records():
		for i in range(1500):
			yield ex(1000 + i, {"x": i})
		raise OSError("disk full")

	with pytest.raises(OSError):
		store.import_records(records(), "replace")
	assert store.count() == 300 and store.get("e0")["output"] == {"x": 0}
