"""Decontamination: held-out or benchmark inputs that nearly repeat a training input make scores look better
than the model is. Texts are compared as sets of word pairs (Jaccard similarity); prefix filtering keeps the
search linear in the training set, so it scales to hundreds of thousands of examples.
"""

import math
import re

THRESHOLD = 0.8
WORD = re.compile(r"\w+")


def shingles(text):
	"""Word pairs of the lower-cased text (single words for one-word texts)."""
	words = WORD.findall(str(text or "").lower())
	if len(words) < 2:
		return set(words)
	return {f"{a} {b}" for a, b in zip(words, words[1:])}


def jaccard(a, b):
	shared = len(a & b)
	return shared / (len(a) + len(b) - shared) if a or b else 0.0


class OverlapIndex:
	"""Indexes the texts to protect (held-out or test inputs) and finds near-copies among other texts."""

	def __init__(self, texts, threshold=THRESHOLD):
		self.threshold = threshold
		self.sets = [shingles(t) for t in texts]
		self.freq = {}
		for s in self.sets:
			for item in s:
				self.freq[item] = self.freq.get(item, 0) + 1
		self.postings = {}
		for i, s in enumerate(self.sets):
			for item in self.prefix(s):
				self.postings.setdefault(item, []).append(i)

	def prefix(self, items):
		"""Two sets at least `threshold` similar share an element among each one's rarest few."""
		ordered = sorted(items, key=lambda x: (self.freq.get(x, 0), x))
		return ordered[:len(ordered) - math.ceil(self.threshold * len(ordered) - 1e-9) + 1]

	def matches(self, text):
		"""[(index of an indexed text, similarity)] at or above the threshold."""
		items = shingles(text)
		if not items:
			return []
		candidates = {i for item in self.prefix(items) for i in self.postings.get(item, ())}
		found = [(i, jaccard(items, self.sets[i])) for i in candidates]
		return sorted(((i, round(sim, 4)) for i, sim in found if sim >= self.threshold), key=lambda x: -x[1])


def find_overlaps(protected, others, threshold=THRESHOLD, limit=200):
	"""Near-duplicate pairs between protected rows (held-out or test) and other rows (training), as
	{"count": protected rows with a near copy, "pairs": [{"held_out", "train", "similarity"}] (at most limit)}.
	Rows are {"id", "input"}; rows with an image are compared by text only when they have text."""
	protected = [r for r in protected if str(r.get("input") or "").strip()]
	index = OverlapIndex([r["input"] for r in protected], threshold)
	pairs, hit = [], set()
	for row in others:
		for i, similarity in index.matches(row.get("input")):
			hit.add(i)
			if len(pairs) < limit:
				pairs.append({"held_out": protected[i].get("id"), "train": row.get("id"), "similarity": similarity})
	return {"threshold": threshold, "count": len(hit), "held_out_ids": sorted(str(protected[i].get("id")) for i in hit), "pairs": pairs}
