import random

import pytest

from llmzempic import decisions
from llmzempic.decision_engine import DecisionFeatures, DecisionReader, anchor_positions, training_items
from test_prompts import FakeTokenizer, decode

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class PadTokenizer(FakeTokenizer):
	pad_token_id = 0


def tiny_model():
	torch.manual_seed(0)
	config = transformers.LlamaConfig(
		vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
		num_key_value_heads=4, max_position_embeddings=1024,
	)
	return transformers.LlamaForCausalLM(config).eval()


def naive_scores(model, tokenizer, reader, state, question):
	"""Each option's log-probability from its own unpadded forward pass."""
	ids = reader.prompt_ids(state, question)
	out = []
	for path in reader.paths(len(question["options"])):
		with torch.no_grad():
			logp = torch.log_softmax(model(torch.tensor([ids + path])).logits[0].float(), -1)
		out.append(sum(logp[len(ids) - 1 + k, t].item() for k, t in enumerate(path)))
	return out


def test_batched_readout_matches_unbatched_forward_passes():
	model, tokenizer = tiny_model(), PadTokenizer()
	reader = DecisionReader(model, tokenizer, "Triage.", batch_rows=3)
	assert reader.keep_arg == "logits_to_keep" and reader.pass_positions
	questions = [
		{"name": "many", "type": "choice", "instructions": "Pick one", "options": [f"option {i}" for i in range(12)]},
		{"name": "ok", "type": "predicate", "instructions": "Is it fine?", "options": [True, False]},
		{"name": "level", "type": "score", "instructions": "", "options": ["low", "high"]},
	]
	scores, tokens = reader.score("a short input", questions)
	assert tokens > 0
	for q, got in zip(questions, scores):
		assert got == pytest.approx(naive_scores(model, tokenizer, reader, "a short input", q), abs=1e-4)


def test_training_features_teach_the_gold_option_number():
	tokenizer = PadTokenizer()
	q = {"name": "pick", "type": "choice", "instructions": "", "options": ["red", "green", "blue"]}
	spec = {"questions": [q], "system_prompt": decisions.system_prompt("Colors.")}
	rows = [{"input": "the sky", "output": {"pick": "blue"}}, {"input": "?", "output": {}}]
	items, lengths = training_items(rows, tokenizer, spec, lambda done, total: None)
	assert items == [(0, 0, 2)] and len(lengths) == 1
	features = DecisionFeatures(rows, items * 8, tokenizer, spec, seed=1)
	seen = set()
	for feature in features:
		prompt = decode([t for t, label in zip(feature["input_ids"], feature["labels"]) if label == -100])
		answer = decode([t for t in feature["labels"] if t != -100])
		number = answer[:1]
		assert f"\n{number}. blue\n" in prompt
		seen.add(number)
	assert len(seen) > 1  # the options were shuffled


def test_anchor_positions_only_predict_prompt_tokens():
	labels = torch.tensor([[-100, -100, -100, -100, 5, 6], [-100, -100, 7, -100, -100, -100]])
	mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
	rows, cols = anchor_positions(labels, mask, 2, torch)
	assert list(zip(rows.tolist(), cols.tolist())) == [(0, 0), (0, 2), (1, 0)]
