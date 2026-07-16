import base64
import json
import os
import tempfile
import unittest
import urllib.parse

import numpy as np

from asr_service.backends import AliyunASRBackend, BaiduASRBackend, XFYunASRBackend, _build_whisper_generation_kwargs


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


class XFYunASRBackendTest(unittest.TestCase):
    def test_transcribe_uploads_file_and_polls_result(self) -> None:
        requests = []
        order_result = json.dumps(
            {
                "lattice": [
                    {
                        "json_1best": json.dumps(
                            {
                                "st": {
                                    "bg": "0",
                                    "ed": "1200",
                                    "rt": [
                                        {
                                            "ws": [
                                                {"cw": [{"w": "你好"}]},
                                                {"cw": [{"w": "世界"}]},
                                            ]
                                        }
                                    ],
                                }
                            }
                        )
                    }
                ]
            }
        )
        responses = [
            {"code": "000000", "content": {"orderId": "order-1"}},
            {
                "code": "000000",
                "content": {
                    "orderInfo": {"orderId": "order-1", "status": 4, "failType": 0},
                    "orderResult": order_result,
                },
            },
        ]

        def fake_urlopen(request, timeout: float):
            requests.append((request, timeout))
            return _FakeHTTPResponse(responses.pop(0))

        backend = XFYunASRBackend(
            {
                "provider": "xfyun",
                "xfyun_api_base_url": "https://raasr.example.com/v2",
                "xfyun_app_id": "app-id",
                "xfyun_access_key_id": "access-key-id",
                "xfyun_access_key_secret": "access-key-secret",
                "xfyun_poll_interval_seconds": 0,
                "xfyun_max_poll_seconds": 1,
            },
            urlopen=fake_urlopen,
            sleep=lambda seconds: None,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(b"fake audio")
            tmp.flush()
            result = backend.transcribe(
                tmp.name,
                language="zh",
                beam_size=5,
                vad_filter=False,
                initial_prompt="",
            )

        self.assertEqual(result.text, "你好世界")
        self.assertEqual(result.duration, 1.2)
        self.assertEqual(len(result.segments), 1)

        upload_request, upload_timeout = requests[0]
        self.assertEqual(upload_timeout, 300)
        self.assertEqual(upload_request.get_full_url().split("?")[0], "https://raasr.example.com/v2/upload")
        self.assertEqual(upload_request.data, b"fake audio")
        upload_headers = dict(upload_request.header_items())
        self.assertIn("Signature", upload_headers)
        self.assertEqual(upload_headers["Content-type"], "application/octet-stream")
        upload_query = urllib.parse.parse_qs(urllib.parse.urlparse(upload_request.get_full_url()).query)
        self.assertEqual(upload_query["appId"], ["app-id"])
        self.assertEqual(upload_query["accessKeyId"], ["access-key-id"])
        self.assertEqual(upload_query["uploadMode"], ["fileStream"])
        self.assertEqual(upload_query["language"], ["autodialect"])
        self.assertEqual(upload_query["durationCheckDisable"], ["true"])

        result_request, _ = requests[1]
        self.assertEqual(result_request.get_full_url().split("?")[0], "https://raasr.example.com/v2/getResult")
        result_query = urllib.parse.parse_qs(urllib.parse.urlparse(result_request.get_full_url()).query)
        self.assertEqual(result_query["orderId"], ["order-1"])
        self.assertEqual(result_query["resultType"], ["transfer"])

    def test_credentials_can_come_from_environment(self) -> None:
        old_values = {key: os.environ.get(key) for key in ("XFYUN_APP_ID", "XFYUN_API_KEY", "XFYUN_API_SECRET")}
        os.environ["XFYUN_APP_ID"] = "env-app-id"
        os.environ["XFYUN_API_KEY"] = "env-api-key"
        os.environ["XFYUN_API_SECRET"] = "env-api-secret"
        try:
            backend = XFYunASRBackend(
                {"provider": "xfyun", "xfyun_api_base_url": "https://example.com/v2"},
                urlopen=lambda request, timeout: _FakeHTTPResponse({"code": "000000"}),
                sleep=lambda seconds: None,
            )
            self.assertEqual(backend.app_id, "env-app-id")
            self.assertEqual(backend.access_key_id, "env-api-key")
            self.assertEqual(backend.access_key_secret, "env-api-secret")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class BaiduASRBackendTest(unittest.TestCase):
    def test_transcribe_fetches_token_and_posts_wav_chunks(self) -> None:
        requests = []
        responses = [
            {"access_token": "access-token", "expires_in": 3600},
            {"err_no": 0, "err_msg": "success.", "result": ["第一段"]},
            {"err_no": 0, "err_msg": "success.", "result": ["第二段"]},
        ]

        def fake_urlopen(request, timeout: float):
            requests.append((request, timeout))
            return _FakeHTTPResponse(responses.pop(0))

        def fake_converter(audio_path: str, rate: int, chunk_seconds: float):
            self.assertEqual(audio_path, "/tmp/input.mp3")
            self.assertEqual(rate, 16000)
            self.assertEqual(chunk_seconds, 55)
            yield b"chunk-one", 1.5, rate
            yield b"chunk-two", 0.75, rate

        backend = BaiduASRBackend(
            {
                "provider": "baidu",
                "baidu_token_url": "https://aip.example.com/oauth/2.0/token",
                "baidu_api_base_url": "https://vop.example.com",
                "baidu_api_key": "api-key",
                "baidu_secret_key": "secret-key",
                "baidu_cuid": "test-cuid",
                "baidu_dev_pid": 1537,
                "baidu_rate": 16000,
                "baidu_channel": 1,
                "baidu_api_timeout_seconds": 12,
                "baidu_chunk_seconds": 55,
            },
            urlopen=fake_urlopen,
            audio_converter=fake_converter,
        )

        result = backend.transcribe(
            "/tmp/input.mp3",
            language="zh",
            beam_size=5,
            vad_filter=False,
            initial_prompt="",
        )

        self.assertEqual(result.text, "第一段\n第二段")
        self.assertEqual(result.duration, 2.25)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].start, 0.0)
        self.assertEqual(result.segments[0].end, 1.5)
        self.assertEqual(result.segments[1].start, 1.5)
        self.assertEqual(result.segments[1].end, 2.25)

        token_request, token_timeout = requests[0]
        self.assertEqual(token_timeout, 12)
        self.assertEqual(token_request.get_method(), "POST")
        token_url = urllib.parse.urlparse(token_request.get_full_url())
        self.assertEqual(token_url.scheme + "://" + token_url.netloc + token_url.path, "https://aip.example.com/oauth/2.0/token")
        token_query = urllib.parse.parse_qs(token_url.query)
        self.assertEqual(token_query["grant_type"], ["client_credentials"])
        self.assertEqual(token_query["client_id"], ["api-key"])
        self.assertEqual(token_query["client_secret"], ["secret-key"])

        first_asr_request, first_asr_timeout = requests[1]
        self.assertEqual(first_asr_timeout, 12)
        self.assertEqual(first_asr_request.get_full_url(), "https://vop.example.com/server_api")
        first_body = json.loads(first_asr_request.data.decode("utf-8"))
        self.assertEqual(first_body["format"], "wav")
        self.assertEqual(first_body["rate"], 16000)
        self.assertEqual(first_body["channel"], 1)
        self.assertEqual(first_body["cuid"], "test-cuid")
        self.assertEqual(first_body["token"], "access-token")
        self.assertEqual(first_body["dev_pid"], 1537)
        self.assertEqual(first_body["len"], len(b"chunk-one"))
        self.assertEqual(base64.b64decode(first_body["speech"]), b"chunk-one")

        second_body = json.loads(requests[2][0].data.decode("utf-8"))
        self.assertEqual(second_body["token"], "access-token")
        self.assertEqual(base64.b64decode(second_body["speech"]), b"chunk-two")

    def test_credentials_can_come_from_environment(self) -> None:
        old_values = {key: os.environ.get(key) for key in ("BAIDU_ASR_API_KEY", "BAIDU_ASR_SECRET_KEY")}
        os.environ["BAIDU_ASR_API_KEY"] = "env-api-key"
        os.environ["BAIDU_ASR_SECRET_KEY"] = "env-secret-key"
        try:
            backend = BaiduASRBackend(
                {"provider": "baidu", "baidu_api_base_url": "https://vop.example.com"},
                urlopen=lambda request, timeout: _FakeHTTPResponse({"access_token": "token"}),
                audio_converter=lambda audio_path, rate, chunk_seconds: [],
            )
            self.assertEqual(backend.api_key, "env-api-key")
            self.assertEqual(backend.secret_key, "env-secret-key")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
