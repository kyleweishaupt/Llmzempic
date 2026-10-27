"""Importing labeled examples from a Hugging Face dataset (BANKING77, CLINC150, any classification set) through
the dataset viewer API, so nothing but the standard library is needed. Datasets with a loading script have no
viewer rows; the API says so and the error is passed on.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co"
PAGE = 100
MAX_ROWS = 50_000
TEXT_HINTS = ("text", "sentence", "utterance", "query", "question", "content", "input", "review", "message")


class HubError(Exception):
	pass


def fetch_json(path, params, token=None, attempts=3):
	url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
	request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
	try:
		with urllib.request.urlopen(request, timeout=60) as response:
			return json.load(response)
	except urllib.error.HTTPError as e:
		if e.code >= 500 and attempts > 1:  # the viewer's gateway fails now and then; try again
			time.sleep(2)
			return fetch_json(path, params, token, attempts - 1)
		try:
			detail = json.load(e).get("error")
		except ValueError:
			detail = None
		raise HubError(detail or f"The dataset viewer answered {e.code}.")
	except (urllib.error.URLError, TimeoutError) as e:
		raise HubError(f"Couldn't reach Hugging Face: {e}")


def column_kind(feature):
	"""text, label (a class label with names), bool, number, or other."""
	kind = feature.get("_type")
	if kind == "ClassLabel":
		return "label"
	if kind == "Value":
		dtype = feature.get("dtype", "")
		if dtype in ("string", "large_string"):
			return "text"
		if dtype == "bool":
			return "bool"
		if dtype.startswith(("int", "uint", "float")):
			return "number"
	return "other"


def describe(dataset, token=None, fetch=fetch_json):
	"""Configs, splits and columns of a dataset, with the likely text and label columns."""
	try:
		info = fetch("info", {"dataset": dataset}, token).get("dataset_info") or {}
	except HubError as e:
		if "config names" in str(e):
			raise HubError(f"{e} Datasets with a loading script can't be read; look for a parquet copy (mteb/ has many).")
		raise
	if not info:
		raise HubError("The dataset viewer has no information about this dataset.")
	configs = []
	for name, config in info.items():
		features = config.get("features") or {}
		columns = [{
			"name": col, "kind": column_kind(f),
			**({"names": f.get("names")} if f.get("_type") == "ClassLabel" else {}),
		} for col, f in features.items()]
		texts = [c["name"] for c in columns if c["kind"] == "text"]
		text = next((c for c in texts if c.lower() in TEXT_HINTS), texts[0] if texts else None)
		labels = [c["name"] for c in columns if c["kind"] in ("label", "bool")] or [c for c in texts if c != text and "label" in c.lower()]
		configs.append({
			"name": name,
			"splits": [{"name": s, "rows": v.get("num_examples")} for s, v in (config.get("splits") or {}).items()],
			"columns": columns,
			"text": text,
			"labels": labels,
		})
	return {"dataset": dataset, "configs": configs}


def iter_rows(dataset, config, split, limit, token=None, fetch=fetch_json, on_page=None):
	"""(features, row) pairs from the viewer, PAGE rows per request, at most limit rows."""
	offset = 0
	while offset < limit:
		page = fetch("rows", {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": min(PAGE, limit - offset)}, token)
		rows = page.get("rows") or []
		features = {f["name"]: f["type"] for f in page.get("features") or []}
		for item in rows:
			yield features, item["row"]
		offset += len(rows)
		if on_page:
			on_page(offset, min(limit, page.get("num_rows_total") or limit))
		if len(rows) < PAGE or offset >= (page.get("num_rows_total") or 0):
			return


def label_value(feature, value):
	if feature and feature.get("_type") == "ClassLabel" and isinstance(value, int):
		names = feature.get("names") or []
		return names[value] if 0 <= value < len(names) else None
	return value


def import_examples(dataset, config, split, text_column, label_columns, limit=5000, token=None, fetch=fetch_json, on_page=None):
	"""(examples {"input", "output": {label column: value}}, values seen per label column, rows skipped,
	column features). Class labels become their names and numbers text; rows with an empty text or a missing
	label are skipped."""
	if not label_columns:
		raise HubError("Choose at least one label column.")
	limit = max(1, min(int(limit), MAX_ROWS))
	examples, seen, skipped, features = [], {c: [] for c in label_columns}, 0, {}
	for features, row in iter_rows(dataset, config, split, limit, token, fetch, on_page):
		text = row.get(text_column)
		output = {}
		for column in label_columns:
			value = label_value(features.get(column), row.get(column))
			if value is None or (isinstance(value, str) and not value.strip()):
				break
			if not isinstance(value, (str, bool)):
				value = str(value)
			output[column] = value
			if value not in seen[column]:
				seen[column].append(value)
		else:
			if isinstance(text, str) and text.strip():
				examples.append({"input": text, "output": output})
				continue
		skipped += 1
	return examples, seen, skipped, features


def labels_schema(seen, features=None):
	"""A decision schema for the label columns: an option list per column (class names in their order), or a
	yes/no field for boolean columns."""
	properties = {}
	for column, values in seen.items():
		feature = (features or {}).get(column) or {}
		if all(isinstance(v, bool) for v in values) and values:
			properties[column] = {"type": "boolean"}
		else:
			names = feature.get("names") if feature.get("_type") == "ClassLabel" else None
			options = [n for n in names if n in values] if names else sorted({str(v) for v in values})
			properties[column] = {"type": "string", "enum": options}
	return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
