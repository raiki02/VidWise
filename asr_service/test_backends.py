import unittest

from asr_service.backends import _build_whisper_generation_kwargs


class _FakeConfig:
    def __init__(self, max_target_positions: int = 448) -> None:
        self.max_target_positions = max_target_positions


class _FakeModel:
    def __init__(self, max_target_positions: int = 448) -> None:
        self.config = _FakeConfig(max_target_positions)


class _FakeTokenizer:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(self.token_count))


class _FakeProcessor:
    def __init__(self, prompt_token_count: int = 0) -> None:
        self.tokenizer = _FakeTokenizer(prompt_token_count)

    def get_decoder_prompt_ids(self, language: str, task: str) -> list[tuple[int, int]]:
        return [(1, 50260), (2, 50359)]


class WhisperGenerationKwargsTest(unittest.TestCase):
    def test_caps_max_new_tokens_by_decoder_prefix(self) -> None:
        kwargs = _build_whisper_generation_kwargs(
            _FakeModel(),
            _FakeProcessor(),
            language="zh",
            beam_size=5,
            initial_prompt="",
        )

        self.assertEqual(kwargs["max_new_tokens"], 445)
        self.assertEqual(len(kwargs["forced_decoder_ids"]), 2)
        self.assertEqual(kwargs["num_beams"], 5)

    def test_caps_prompt_and_max_new_tokens_within_model_limit(self) -> None:
        kwargs = _build_whisper_generation_kwargs(
            _FakeModel(),
            _FakeProcessor(prompt_token_count=300),
            language="zh",
            beam_size=1,
            initial_prompt="domain words",
        )

        self.assertEqual(len(kwargs["prompt_ids"]), 224)
        self.assertEqual(kwargs["max_new_tokens"], 221)
        self.assertEqual(3 + len(kwargs["prompt_ids"]) + kwargs["max_new_tokens"], 448)
        self.assertNotIn("num_beams", kwargs)

    def test_trims_prompt_to_leave_generation_room_on_small_models(self) -> None:
        kwargs = _build_whisper_generation_kwargs(
            _FakeModel(max_target_positions=16),
            _FakeProcessor(prompt_token_count=100),
            language="zh",
            beam_size=1,
            initial_prompt="domain words",
        )

        self.assertLessEqual(3 + len(kwargs["prompt_ids"]) + kwargs["max_new_tokens"], 16)
        self.assertGreaterEqual(kwargs["max_new_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
