import sqlite3

import pytest

from llmzempic.datasets import DataSetError, Library, iter_values, preview


def write(tmp_path, name, text):
	path = tmp_path / name
	path.write_text(text, encoding="utf-8")
	return path


@pytest.fixture
def library(tmp_path):
	return Library(tmp_path / "datasets")


def test_csv_auto_picks_value_and_description_columns(tmp_path):
	text = "id,description,extra\n  V1 , First value ,x\nV2,Second,y\n"
	info = preview(text, "values.csv")
	assert info["format"] == "csv"
	assert info["columns"] == ["id", "description", "extra"]
	assert (info["value_column"], info["description_column"]) == ("id", "description")
	path = write(tmp_path, "values.csv", text)
	assert list(iter_values(path, "values.csv")) == [("V1", "First value"), ("V2", "Second")]


def test_txt_value_then_description(tmp_path):
	path = write(tmp_path, "codes.txt", "A00.0  Cholera\nB01.1\tChickenpox\nC02\n")
	assert list(iter_values(path, "codes.txt")) == [("A00.0", "Cholera"), ("B01.1", "Chickenpox"), ("C02", None)]


def test_duplicates_collapse_and_are_counted(tmp_path, library):
	path = write(tmp_path, "v.txt", "V1\nV2\nV1\nV1\nV3\n")
	meta = library.create("Codes", path, "v.txt")
	assert meta["rows"] == 3
	assert meta["duplicates"] == 2


def test_missing_is_exact(tmp_path, library):
	meta = library.create("Codes", write(tmp_path, "v.txt", "V1\n"), "v.txt")
	assert library.get(meta["id"]).missing(["V1", "v1", " V1", "V2"]) == {"v1", " V1", "V2"}


def test_search_prefers_value_prefix_then_description_words(tmp_path, library):
	text = "A10  Fracture of arm\nB20  Arm sprain\nA11  Headache\nC30  Leg fracture\n"
	meta = library.create("Codes", write(tmp_path, "v.txt", text), "v.txt")
	ds = library.get(meta["id"])
	assert [r["value"] for r in ds.search("A1")] == ["A10", "A11"]
	assert {r["value"] for r in ds.search("fracture")} == {"A10", "C30"}
	assert ds.search("arm")[0]["value"] in ("A10", "B20")
	assert ds.search("")[:1] and len(ds.search("", limit=2)) == 2


def test_same_content_reuses_existing_set(tmp_path, library):
	first = library.create("Codes", write(tmp_path, "a.txt", "V1  one\nV2  two\n"), "a.txt")
	second = library.create("Other name", write(tmp_path, "b.txt", "V2  two\nV1  one\n"), "b.txt")
	assert second["hash"] == first["hash"]
	assert second["id"] == first["id"] and second["reused"] is True
	assert [m["id"] for m in library.list()] == [first["id"]]


def test_unusable_file_raises(tmp_path, library):
	with pytest.raises(DataSetError):
		library.create("Empty", write(tmp_path, "e.txt", "\n  \n"), "e.txt")
	with pytest.raises(DataSetError):
		library.create("Object", write(tmp_path, "o.json", '{"a": 1}'), "o.json")
	assert library.list() == []


def test_replace_changes_hash_and_delete_removes(tmp_path, library):
	meta = library.create("Codes", write(tmp_path, "a.txt", "V1\n"), "a.txt")
	replaced = library.replace(meta["id"], write(tmp_path, "b.txt", "V1\nV2\n"), "b.txt")
	assert replaced["id"] == meta["id"] and replaced["hash"] != meta["hash"] and replaced["rows"] == 2
	assert library.by_hash(replaced["hash"]).meta["id"] == meta["id"]
	library.delete(meta["id"])
	with pytest.raises(KeyError):
		library.get(meta["id"])


def test_preview_honours_chosen_columns():
	text = "code,label,notes\nA,Alpha,first\nB,Beta,second\n"
	info = preview(text, "x.csv", value_column="label", description_column="notes")
	assert (info["value_column"], info["description_column"]) == ("label", "notes")
	assert info["rows"][0] == {"value": "Alpha", "description": "first"}
	assert preview(text, "x.csv", value_column="code", description_column="")["rows"][0] == {"value": "A", "description": None}


def test_values_with_line_breaks_are_refused(tmp_path, library):
	# Exports ship one value per line, so a value can't contain one.
	path = write(tmp_path, "values.csv", 'code,description\n"A\nB",x\nC,y\n')
	with pytest.raises(DataSetError, match="line break"):
		library.create("Bad", path, "values.csv")


def test_replace_while_the_values_are_open(tmp_path, library):
	# Checks and searches keep the set's database open; on Windows an open file can't be moved.
	library.create("Codes", write(tmp_path, "a.txt", "A\nB\n"), "a.txt")
	reader = sqlite3.connect(str(library.get("codes").path / "values.db"))
	try:
		assert reader.execute("SELECT count(*) FROM rows").fetchone()[0] == 2
		library.replace("codes", write(tmp_path, "b.txt", "C\n"), "b.txt")
	finally:
		reader.close()
	assert [m["id"] for m in library.list()] == ["codes"]
	assert library.get("codes").missing(["A", "C"]) == {"A"}
	assert library.by_hash(library.get("codes").meta["hash"]).meta["rows"] == 1
