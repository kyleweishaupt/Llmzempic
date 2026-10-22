"""Feeding a text run's examples to the trainer without holding them all in memory."""

from .prompts import tokenize_example


def select_within(lengths, max_len):
	"""(indices to train on, indices skipped for being longer than max_len tokens); 0 keeps all."""
	keep, skipped = [], []
	for i, length in enumerate(lengths):
		(skipped if max_len and length > max_len else keep).append(i)
	return keep, skipped


def measure_lengths(tokenizer, system_prompt, rows, batch, on_progress):
	"""[(tokens in the example, tokens in its answer)] for every row, reporting progress every batch rows."""
	lengths = []
	for i, row in enumerate(rows, start=1):
		feature = tokenize_example(tokenizer, system_prompt, row["input"], row["output"])
		lengths.append((len(feature["input_ids"]), sum(1 for t in feature["labels"] if t != -100)))
		if i % batch == 0:
			on_progress(i, len(rows))
	on_progress(len(lengths), len(rows))
	return lengths


class LazyFeatures:
	"""The selected rows, tokenized when the trainer asks for them."""

	def __init__(self, rows, indices, tokenizer, system_prompt):
		self.rows = rows
		self.indices = list(indices)
		self.tokenizer = tokenizer
		self.system_prompt = system_prompt

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, i):
		row = self.rows[self.indices[i]]
		return tokenize_example(self.tokenizer, self.system_prompt, row["input"], row["output"])
