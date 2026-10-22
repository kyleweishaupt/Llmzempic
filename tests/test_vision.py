import pytest

from llmzempic import vision


class FakeImageProcessor:
	pass


class FakeProcessor:
	"""Chat template that rejects the system role, like some vision templates."""

	def __init__(self, allow_system=True):
		self.allow_system = allow_system
		self.image_processor = FakeImageProcessor()

	def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
		if not self.allow_system and messages[0]["role"] == "system":
			raise ValueError("System role not supported")
		return messages


def test_messages_put_the_image_first_and_skip_empty_text():
	messages = vision.build_messages("SYS", "  ", answer='{"hot_dog":true}')
	assert [m["role"] for m in messages] == ["system", "user", "assistant"]
	assert messages[1]["content"] == [{"type": "image"}]
	with_text = vision.build_messages("", "On a plate")
	assert with_text == [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "On a plate"}]}]


def test_system_prompt_folds_into_user_turn_when_template_rejects_it():
	messages = vision.render(FakeProcessor(allow_system=False), vision.build_messages("SYS", "hi"), True)
	assert messages == [{"role": "user", "content": [
		{"type": "image"}, {"type": "text", "text": "SYS"}, {"type": "text", "text": "hi"},
	]}]


def test_resolution_is_capped_per_processor_family():
	qwen = FakeProcessor()
	qwen.image_processor.merge_size, qwen.image_processor.patch_size = 2, 16
	qwen.image_processor.size = {"shortest_edge": 65536, "longest_edge": 16777216}
	vision.limit_resolution(qwen)
	assert qwen.image_processor.size == {"shortest_edge": 65536, "longest_edge": vision.MAX_IMAGE_PIXELS}
	assert vision.image_max_tokens(qwen) == 256

	smol = FakeProcessor()
	smol.image_processor.do_image_splitting = True
	smol.image_processor.max_image_size = {"longest_edge": 512}
	smol.image_processor.size = {"longest_edge": 2048}
	smol.image_seq_len = 64
	vision.limit_resolution(smol)
	assert smol.image_processor.do_image_splitting is False
	assert smol.image_processor.size == {"longest_edge": 512}
	assert vision.image_max_tokens(smol) == 64


def test_load_image_applies_exif_rotation(tmp_path):
	Image = pytest.importorskip("PIL.Image")
	sideways = Image.new("RGB", (40, 20), (200, 30, 30))
	exif = Image.Exif()
	exif[0x0112] = 6  # stored rotated; viewers turn it upright
	sideways.save(tmp_path / "photo.jpg", exif=exif)
	image = vision.load_image(tmp_path / "photo.jpg")
	assert image.size == (20, 40)
	assert image.mode == "RGB"


def test_collator_masks_prompt_and_padding(monkeypatch):
	torch = pytest.importorskip("torch")
	monkeypatch.setattr(vision, "load_image", lambda path: None)

	class Tokenizer:
		padding_side = "left"

	class Processor:
		tokenizer = Tokenizer()

		def __call__(self, text, images, padding, return_tensors):
			assert self.tokenizer.padding_side == "right"
			lengths = [len(t) for t in text]
			longest = max(lengths)
			ids = torch.tensor([[1] * n + [0] * (longest - n) for n in lengths])
			return {"input_ids": ids, "attention_mask": (ids != 0).long(), "pixel_values": torch.zeros(len(text), 3), "token_type_ids": ids}

	class Model:
		def forward(self, input_ids, attention_mask, pixel_values, labels):
			pass

	collator = vision.VisionCollator(Processor(), Model())
	batch = collator([{"text": "abcdef", "image": "x", "prompt_len": 4}, {"text": "abc", "image": "y", "prompt_len": 2}])
	assert set(batch) == {"input_ids", "attention_mask", "pixel_values", "labels"}
	assert batch["labels"].tolist() == [[-100, -100, -100, -100, 1, 1], [-100, -100, 1, -100, -100, -100]]
