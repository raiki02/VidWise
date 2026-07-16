import json
import os
import unittest

import numpy as np

from asr_service.backends import AliyunASRBackend, _build_whisper_generation_kwargs


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


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class AliyunASRBackendTest(unittest.TestCase):
    def test_transcribe_posts_audio_to_openai_compatible_endpoint(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append((request, timeout))
            return _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "测试文本",
                                "annotations": [{"language": "zh"}],
                            }
                        }
                    ],
                    "usage": {"seconds": 1.25},
                }
            )

        backend = AliyunASRBackend(
            {
                "provider": "aliyun",
                "name": "qwen3-asr-flash",
                "api_base_url": "https://dashscope.example.com/compatible-mode/v1",
                "api_key": "sk-test",
            },
            urlopen=fake_urlopen,
        )

        result = backend.transcribe(
            np.zeros(1600, dtype=np.float32),
            language="zh",
            beam_size=5,
            vad_filter=False,
            initial_prompt="领域词",
            sample_rate=16000,
        )

        self.assertEqual(result.text, "测试文本")
        self.assertEqual(result.language, "zh")
        self.assertEqual(result.duration, 1.25)
        self.assertEqual(len(result.segments), 1)
        request, timeout = requests[0]
        self.assertEqual(timeout, 300)
        self.assertEqual(request.get_full_url(), "https://dashscope.example.com/compatible-mode/v1/chat/completions")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer sk-test")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3-asr-flash")
        self.assertEqual(body["messages"][0]["role"], "system")
        audio_data = body["messages"][1]["content"][0]["input_audio"]["data"]
        self.assertTrue(audio_data.startswith("data:audio/wav;base64,"))
        self.assertEqual(body["asr_options"]["language"], "zh")

    def test_api_key_can_come_from_environment(self) -> None:
        old_value = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "sk-env"
        try:
            backend = AliyunASRBackend(
                {"provider": "aliyun", "api_base_url": "https://example.com/v1"},
                urlopen=lambda request, timeout: _FakeHTTPResponse({"choices": [{"message": {"content": ""}}]}),
            )
            self.assertEqual(backend.api_key, "sk-env")
        finally:
            if old_value is None:
                os.environ.pop("DASHSCOPE_API_KEY", None)
            else:
                os.environ["DASHSCOPE_API_KEY"] = old_value


if __name__ == "__main__":
    unittest.main()
