"""Write a synthetic large project for scale testing: a data set, examples that use it, and a schema.

    python tools/make_scale_fixture.py --examples 200000 --values 70000 --out <dir>

Then upload <dir>/values.csv on the Data sets page as "Scale values", create a text project, paste
<dir>/schema.json into the Task tab's JSON view, and import <dir>/examples.jsonl on the Examples tab.
"""

import argparse
import csv
import json
import random
from pathlib import Path

WORDS = (
	"report note review item history finding result follow visit plan current prior noted "
	"stable change level normal mild strong sharp steady left right upper lower daily weekly "
	"started stopped continue monitor assess treat order sample record status measure value "
	"score trend increase decrease without with after before during since onset course"
).split()


def sentence(rng):
	words = rng.choices(WORDS, k=rng.randint(10, 24))
	return " ".join(words).capitalize() + "."


def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--examples", type=int, default=200_000)
	parser.add_argument("--values", type=int, default=70_000)
	parser.add_argument("--chars", type=int, default=5000, help="Approximate input length")
	parser.add_argument("--dataset-id", default="scale-values", help="Id the data set gets when uploaded")
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--out", required=True)
	args = parser.parse_args()

	rng = random.Random(args.seed)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)

	codes = [f"{chr(65 + i % 26)}{i // 26:04d}" for i in range(args.values)]
	descriptions = [" ".join(rng.choices(WORDS, k=4)) for _ in codes]
	with open(out / "values.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["code", "description"])
		writer.writerows(zip(codes, descriptions))

	sentences = [sentence(rng) for _ in range(4000)]
	per_input = max(1, args.chars // 120)
	# Cubing a uniform draw skews toward low indexes: a few common values and a long tail, like real labels.
	pick = lambda: int(args.values * rng.random() ** 3)
	with open(out / "examples.jsonl", "w", encoding="utf-8") as f:
		for _ in range(args.examples):
			primary = pick()
			secondary = [i for i in dict.fromkeys(pick() for _ in range(rng.randint(0, 5))) if i != primary]
			body = rng.choices(sentences, k=per_input)
			for i in [primary, *secondary]:
				body.insert(rng.randrange(len(body) + 1), f"Assessed: {descriptions[i]}.")
			output = {"primary": [codes[primary]], "secondary": [codes[i] for i in secondary]}
			f.write(json.dumps({"input": " ".join(body), "output": output}) + "\n")

	ref = {"type": "string", "x-dataset": args.dataset_id}
	schema = {
		"type": "object",
		"properties": {
			"primary": {"type": "array", "items": ref, "minItems": 1, "maxItems": 1},
			"secondary": {"type": "array", "items": ref, "maxItems": 5},
		},
		"required": ["primary", "secondary"],
		"additionalProperties": False,
	}
	(out / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
	print(f"Wrote {args.values:,} values and {args.examples:,} examples to {out}")


if __name__ == "__main__":
	main()
