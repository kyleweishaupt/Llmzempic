from llmzempic.dataset import input_key, split_examples, split_keys
from llmzempic.schema_tools import TEMPLATES

CLASSIFY = next(t for t in TEMPLATES if t["id"] == "classification")["schema"]
CODES = {"type": "object", "properties": {"b": {"type": "array", "items": {"type": "string", "x-dataset": "codes"}}}}


def labeled(n_billing, n_technical):
	labels = ["billing"] * n_billing + ["technical"] * n_technical
	return [{"id": str(i), "input": f"ticket {i}", "output": {"label": label}} for i, label in enumerate(labels)]


def split(rows, schema, fraction, seed, **kwargs):
	"""Split, then map ids back to rows so assertions can look at them."""
	by_id = {r["id"]: r for r in rows}
	train, held_out, info = split_examples(split_keys(rows, schema), schema, fraction, seed, **kwargs)
	return [by_id[i] for i in train], [by_id[i] for i in held_out], info


def ids(rows):
	return sorted(r["id"] for r in rows)


def test_split_is_stratified_by_label():
	for seed in range(5):
		train, held_out, info = split(labeled(70, 30), CLASSIFY, 0.2, seed)
		assert info == {"stratified_by": "label", "groups": 100, "protected": 0, "sampled_out": 0}
		assert len(held_out) == 20 and len(train) == 80
		counts = {label: sum(r["output"]["label"] == label for r in held_out) for label in ("billing", "technical")}
		assert 13 <= counts["billing"] <= 15 and 5 <= counts["technical"] <= 7, counts


def test_split_keeps_copies_together():
	examples = labeled(20, 20)
	for i in range(0, 12, 3):
		examples.append({"id": f"copy{i}", "input": f"  ticket\n{i} ", "output": examples[i]["output"]})
		examples.append({"id": f"conflict{i}", "input": f"ticket {i}", "output": {"label": "other"}})
	for seed in range(10):
		train, held_out, info = split(examples, CLASSIFY, 0.25, seed)
		assert info["groups"] == 40
		assert len(train) + len(held_out) == len(examples)
		train_inputs = {" ".join(r["input"].split()) for r in train}
		assert not any(" ".join(r["input"].split()) in train_inputs for r in held_out)


def test_split_is_deterministic_per_seed():
	examples = labeled(30, 30)
	first = split(examples, CLASSIFY, 0.2, 7)
	assert [r["id"] for r in first[1]] == [r["id"] for r in split(examples, CLASSIFY, 0.2, 7)[1]]
	assert ids(first[1]) != ids(split(examples, CLASSIFY, 0.2, 8)[1])


def test_split_edge_cases():
	assert split(labeled(5, 4), CLASSIFY, 0.5, 1)[1] == []
	assert split(labeled(10, 10), CLASSIFY, 0, 1)[1] == []
	train, held_out, _ = split(labeled(10, 10), CLASSIFY, 1.0, 1)
	assert len(train) >= 1 and len(train) + len(held_out) == 20
	extraction = next(t for t in TEMPLATES if t["id"] == "extraction")["schema"]
	assert split(labeled(10, 10), extraction, 0.1, 1)[2]["stratified_by"] is None
	assert len(split(labeled(10, 10), {"type": "nonsense"}, 0.1, 1)[1]) == 2
	pairs = [{"id": f"{i}{c}", "input": f"t{i}", "output": {"label": "billing"}} for i in range(6) for c in "ab"]
	held_out = split(pairs, CLASSIFY, 0.1, 1)[1]
	assert len(held_out) == 2 and held_out[0]["input"] == held_out[1]["input"]
	big = labeled(20, 20) + [{"id": f"c{i}", "input": "same", "output": {"label": "billing"}} for i in range(10)]
	for seed in range(20):
		assert len(split(big, CLASSIFY, 0.1, seed)[1]) == 5


def test_image_examples_are_the_same_input_only_with_the_same_image_and_text():
	examples = [
		{"id": "1", "input": "", "image": "a.jpg", "output": {"label": "billing"}},
		{"id": "2", "input": " ", "image": "a.jpg", "output": {"label": "technical"}},
		{"id": "3", "input": "", "image": "b.jpg", "output": {"label": "billing"}},
		{"id": "4", "input": "close up", "image": "a.jpg", "output": {"label": "billing"}},
	]
	assert split(examples, CLASSIFY, 0.5, 1)[2]["groups"] == 3


def values_in(rows):
	return {v for r in rows for v in r["output"]["b"]}


def test_rare_values_never_held_out():
	rows = [{"id": f"u{i}", "input": f"rare {i}", "output": {"b": [f"U{i}", "C"]}} for i in range(10)]
	rows += [{"id": f"r{i}", "input": f"common {i}", "output": {"b": [f"P{i % 5}"]}} for i in range(30)]
	for seed in range(10):
		train, held_out, info = split(rows, CODES, 0.5, seed)
		assert len(held_out) == 20
		assert not any(v.startswith("U") for v in values_in(held_out))
		assert values_in(train) == values_in(rows)
		assert info["protected"] == 10


def test_held_out_max_caps():
	rows = [{"id": str(i), "input": f"t{i}", "output": {"label": "billing"}} for i in range(5000)]
	assert len(split(rows, CLASSIFY, 0.5, 1, held_out_max=300)[1]) == 300
	assert len(split(rows, CLASSIFY, 0.5, 1)[1]) == 300


def test_sampling_keeps_every_value():
	rows = [{"id": str(i), "input": f"t{i}", "output": {"b": [f"V{i % 50}"] + ([f"U{i}"] if i < 30 else [])}} for i in range(1000)]
	for seed in range(5):
		train, held_out, info = split(rows, CODES, 0.1, seed, max_examples=100)
		assert len(train) + len(held_out) == 100
		assert info["sampled_out"] == 900
		assert values_in(train) == values_in(rows)


def test_sampling_never_drops_below_the_protected_examples():
	rows = [{"id": str(i), "input": f"t{i}", "output": {"b": [f"U{i}"]}} for i in range(40)]
	train, held_out, info = split(rows, CODES, 0.1, 1, max_examples=10)
	assert len(train) == 40 and held_out == [] and info["sampled_out"] == 0



def test_sampling_leaves_examples_to_score():
	# Most values are rare, so sampling can only drop rows of the common value; it must not drop every
	# row that could be held out, or the run would have nothing to score.
	rows = [{"id": f"u{i}", "input": f"rare {i}", "output": {"b": [f"U{i}"]}} for i in range(90)]
	rows += [{"id": f"c{i}", "input": f"common {i}", "output": {"b": ["C"]}} for i in range(20)]
	for seed in range(5):
		train, held_out, info = split(rows, CODES, 0.1, seed, max_examples=50)
		assert held_out and all(r["id"].startswith("c") for r in held_out)
		assert values_in(train) == values_in(rows)

def test_input_key_is_a_short_digest_that_ignores_whitespace():
	text = "word " * 2000
	assert input_key({"input": text}) == input_key({"input": "  " + text.replace(" ", "\n")})
	assert len(input_key({"input": text})) <= 64
	assert input_key({"input": "  "}) == ""
	assert input_key({"input": "a", "image": "x.png"}) != input_key({"input": "a"})
