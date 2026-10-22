"""Scores for data set fields: exact (ordered) match, first-value accuracy, partial credit
(micro precision/recall/F1 over values), values outside the data set, and a short per-value table
pointing at the values the model gets wrong most."""

from collections import Counter

from .metrics import parse_generated
from .refs import ref_fields, ref_values

TOP_FREQUENT = 15
TOP_WORST = 15


def _values(output, schema, field):
	return [v for loc, dataset_id, v in ref_values(output, schema)
		if loc.split("/")[0] == field["field"] and dataset_id == field["dataset"]]


def _rate(hits, total):
	return round(hits / total, 4) if total else None


def _field_scores(items, schema, field, resolver):
	pairs = []
	for item in items:
		expected = item.get("expected")
		if not isinstance(expected, dict):
			continue
		value, ok = parse_generated(item.get("generated"))
		pairs.append((_values(expected, schema, field), _values(value, schema, field) if ok else []))
	missing = resolver.missing(field["dataset"], {v for _, got in pairs for v in got})
	exact = first = first_total = invalid = tp = predicted_total = expected_total = 0
	per_value = {}
	for want, got in pairs:
		exact += want == got
		if want:
			first_total += 1
			first += got[:1] == want[:1]
		invalid += any(v in missing for v in got)
		overlap = Counter(want) & Counter(got)
		tp += sum(overlap.values())
		predicted_total += len(got)
		expected_total += len(want)
		for v in set(want) | set(got):
			row = per_value.setdefault(v, {"value": v, "expected": 0, "predicted": 0, "correct": 0})
			row["expected"] += want.count(v)
			row["predicted"] += got.count(v)
			row["correct"] += overlap[v]
	precision, recall = _rate(tp, predicted_total), _rate(tp, expected_total)
	f1 = None
	if precision is not None and recall is not None:
		f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
	expected_rows = [r for r in per_value.values() if r["expected"]]
	frequent = sorted(expected_rows, key=lambda r: (-r["expected"], r["value"]))[:TOP_FREQUENT]
	worst = sorted((r for r in expected_rows if r["expected"] >= 2), key=lambda r: (r["correct"] / r["expected"], -r["expected"], r["value"]))[:TOP_WORST]
	table = list({r["value"]: r for r in frequent + worst}.values())
	return {
		"field": field["field"], "dataset": field["dataset"], "name": resolver.name(field["dataset"]),
		"count": len(pairs), "exact": _rate(exact, len(pairs)), "first": _rate(first, first_total),
		"precision": precision, "recall": recall, "f1": f1,
		"invalid_rate": _rate(invalid, len(pairs)), "values": table,
	}


def ref_scores(items, schema, resolver):
	return [_field_scores(items, schema, field, resolver) for field in ref_fields(schema)]
