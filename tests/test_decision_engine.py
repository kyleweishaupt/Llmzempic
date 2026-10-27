import random

import pytest

from llmzempic import decisions
from llmzempic.decision_engine import (
	DecisionCollator, DecisionFeatures, DecisionReader, anchor_positions, answer_paths, choice_steps, listwise_loss, text_completion,
	training_items,
)
from llmzempic.trainer import PadCollator
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


QUESTIONS = [
	{"name": "many", "type": "choice", "instructions": "Pick one", "options": [f"option {i}" for i in range(12)]},
	{"name": "ok", "type": "predicate", "instructions": "Is it fine?", "options": [True, False]},
	{"name": "level", "type": "score", "instructions": "", "options": ["low", "high"]},
]


@pytest.mark.parametrize("cache", [False, True])
def test_batched_readout_matches_unbatched_forward_passes(cache):
	model, tokenizer = tiny_model(), PadTokenizer()
	reader = DecisionReader(model, tokenizer, "Triage.", batch_rows=3, cache=cache)
	assert reader.keep_arg == "logits_to_keep" and reader.pass_positions and reader.cache == cache
	scores, usage = reader.score("a short input that every question shares", QUESTIONS)
	assert usage["input_tokens"] > 0 and (usage["cached_tokens"] > 0) == cache
	assert reader.cache == cache  # the cached read didn't fall back
	for q, got in zip(QUESTIONS, scores):
		assert got == pytest.approx(naive_scores(model, tokenizer, reader, "a short input that every question shares", q), abs=1e-4)


def test_samples_average_reads_in_shuffled_orders():
	model, tokenizer = tiny_model(), PadTokenizer()
	reader = DecisionReader(model, tokenizer)
	q = QUESTIONS[0]
	scores, _ = reader.score("input", [q], samples=3)
	reads = []
	for order in decisions.sample_orders(q, 3):
		ids = reader.prompt_ids("input", q, order)
		read = [0.0] * len(order)
		for pos, path in enumerate(reader.paths(len(order))):
			with torch.no_grad():
				logp = torch.log_softmax(model(torch.tensor([ids + path])).logits[0].float(), -1)
			read[order[pos]] = sum(logp[len(ids) - 1 + k, t].item() for k, t in enumerate(path))
		reads.append(read)
	assert decisions.softmax(scores[0]) == pytest.approx(decisions.softmax(decisions.combine_samples(reads)), abs=1e-4)


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


def test_listwise_steps_follow_the_option_number_tree():
	tokenizer = PadTokenizer()
	paths = answer_paths(text_completion(tokenizer, "sys"), 12)  # "01".."12": two tokens each
	feature = {"input_ids": [5, 6, 7] + paths[10] + [9], "labels": [-100, -100, -100] + paths[10] + [9]}
	steps = choice_steps(feature, paths, 10)
	# First digit: "0" or "1"; after "1": "0", "1" or "2".
	assert steps == [[2, 1, sorted({ord("0"), ord("1")})], [3, 1, [ord("0"), ord("1"), ord("2")]]]
	assert choice_steps({"input_ids": [5, 6], "labels": [-100, 6]}, paths, 0) == []

	q = {"name": "pick", "type": "choice", "instructions": "", "options": ["red", "green", "blue"]}
	spec = {"questions": [q], "system_prompt": "Colors."}
	features = DecisionFeatures([{"input": "sky", "output": {"pick": "blue"}}], [(0, 0, 2)], tokenizer, spec, listwise=True)
	batch = DecisionCollator(PadCollator(0))([features[0], features[0]])
	assert batch["choice_rows"].tolist() == [0, 1] and batch["choice_tokens"].shape == (2, 3)
	logits = torch.randn(2, batch["input_ids"].shape[1], 256)
	loss = listwise_loss(logits, {k: v for k, v in batch.items() if k.startswith("choice_")}, torch.nn.functional)
	pos, gold = batch["choice_pos"][0].item(), batch["choice_gold"][0].item()
	candidates = logits[0, pos, batch["choice_tokens"][0]]
	assert loss.item() == pytest.approx(-torch.log_softmax(candidates, -1)[gold].item() / 2 - torch.log_softmax(logits[1, pos, batch["choice_tokens"][1]], -1)[gold].item() / 2, abs=1e-5)


def test_anchor_positions_only_predict_prompt_tokens():
	labels = torch.tensor([[-100, -100, -100, -100, 5, 6], [-100, -100, 7, -100, -100, -100]])
	mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
	rows, cols = anchor_positions(labels, mask, 2, torch)
	assert list(zip(rows.tolist(), cols.tolist())) == [(0, 0), (0, 2), (1, 0)]


class FakeProcessor:
	"""A vision processor stand-in: each image becomes three tokens of id 1, text is character codes."""

	def __init__(self):
		self.tokenizer = PadTokenizer()

	def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
		text = "".join(
			f"<|{m['role']}|>" + "".join("<image>" if c["type"] == "image" else c["text"] for c in m["content"]) + "<|end|>"
			for m in messages
		)
		return text + ("<|assistant|>" if add_generation_prompt else "")

	def __call__(self, text, images, return_tensors="pt", padding=False):
		rows = []
		for t in text:
			ids = []
			for i, chunk in enumerate(t.split("<image>")):
				ids += ([1, 1, 1] if i else []) + [ord(c) for c in chunk]
			rows.append(ids)
		return {
			"input_ids": torch.tensor(rows),
			"attention_mask": torch.ones(len(rows), len(rows[0]), dtype=torch.long),
			"pixel_values": torch.zeros(len(rows), 3, 2, 2),
		}


def write_image(path):
	Image = pytest.importorskip("PIL.Image")
	Image.new("RGB", (8, 8), (200, 10, 10)).save(path)
	return path


def test_vision_readout_reads_each_question_with_its_image(tmp_path):
	from llmzempic.decision_engine import VisionDecisionReader

	model, processor = tiny_model(), FakeProcessor()
	reader = VisionDecisionReader(model, processor, "Look closely.")
	image = write_image(tmp_path / "red.png")
	with pytest.raises(decisions.DecisionError, match="image"):
		reader.score("x", QUESTIONS)
	scores, usage = reader.score("", QUESTIONS, image=image)
	assert usage["cached_tokens"] == 0
	for q, got in zip(QUESTIONS, scores):
		text = reader.prompt_text("", q)
		assert "Input: the image." in text and text.count("<image>") == 1
		ids = processor(text=[text], images=[[None]])["input_ids"][0].tolist()
		want = []
		for path in reader.paths(len(q["options"])):
			with torch.no_grad():
				logp = torch.log_softmax(model(torch.tensor([ids + path])).logits[0].float(), -1)
			want.append(sum(logp[len(ids) - 1 + k, t].item() for k, t in enumerate(path)))
		assert got == pytest.approx(want, abs=1e-4)


def test_vision_features_count_image_tokens_in_the_prompt(tmp_path):
	from llmzempic.decision_engine import VisionDecisionFeatures, vision_training_items

	write_image(tmp_path / "a.png")
	q = {"name": "red", "type": "predicate", "instructions": "Is it red?", "options": [True, False]}
	spec = {"questions": [q], "system_prompt": "Look."}
	rows = [{"input": "", "image": "a.png", "output": {"red": False}}]
	items, lengths = vision_training_items(rows, FakeProcessor(), spec, tmp_path, lambda done, total: None, print)
	assert items == [(0, 0, 1)]
	feature = VisionDecisionFeatures(rows, items, FakeProcessor(), spec, tmp_path, listwise=True)[0]
	assert feature["length"] == lengths[0] and feature["text"].endswith("2<|end|>")
	ids = FakeProcessor()(text=[feature["text"]], images=[[None]])["input_ids"][0].tolist()
	assert ids[feature["prompt_len"]] == ord("2") and ids.count(1) == 3
	assert feature["choices"] == [[feature["prompt_len"] - 1, 1, [ord("1"), ord("2")]]]


class SpacedTokenizer(PadTokenizer):
	"""A template that writes the assistant's answer after a space, like SmolVLM's and Mistral's."""

	def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
		text = "".join(f"{m['role']}:{' ' + m['content'] if m['role'] == 'assistant' else m['content']}<end>\n" for m in messages)
		return text + ("assistant:" if add_generation_prompt else "")


def test_answer_paths_follow_the_chat_template():
	paths = answer_paths(text_completion(SpacedTokenizer(), "sys"), 3)
	assert paths == [[ord(" "), ord("1")], [ord(" "), ord("2")], [ord(" "), ord("3")]]
	# The reader scores exactly the tokens training puts after the prompt.
	q = {"name": "pick", "type": "choice", "instructions": "", "options": ["a", "b", "c"]}
	feature = DecisionFeatures([{"input": "x", "output": {"pick": "b"}}], [(0, 0, 1)], SpacedTokenizer(), {"questions": [q], "system_prompt": "sys"}, listwise=True)[0]
	assert feature["choices"] == [[feature["labels"].index(ord(" ")), 1, [ord("1"), ord("2"), ord("3")]]]
	model = tiny_model()
	reader = DecisionReader(model, SpacedTokenizer(), "")
	reader.system_prompt = "sys"
	reader.completion = text_completion(SpacedTokenizer(), "sys")
	assert reader.paths(3) == paths
