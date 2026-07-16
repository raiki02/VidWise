import json
import unittest

from embedding_service.backends import AliyunEmbeddingBackend


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class AliyunEmbeddingBackendTest(unittest.TestCase):
    def test_embed_posts_batch_to_openai_compatible_endpoint(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append((request, timeout))
            return _FakeHTTPResponse(
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                }
            )

        backend = AliyunEmbeddingBackend(
            "text-embedding-v4",
            {
                "api_base_url": "https://dashscope.example.com/compatible-mode/v1",
                "api_key": "sk-test",
                "dimensions": 2,
                "batch_size": 10,
            },
            urlopen=fake_urlopen,
        )

        embeddings = backend.embed(["alpha", "beta"])

        self.assertEqual(embeddings, [[1.0, 0.0], [0.0, 1.0]])
        request, timeout = requests[0]
        self.assertEqual(timeout, 120)
        self.assertEqual(request.get_full_url(), "https://dashscope.example.com/compatible-mode/v1/embeddings")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer sk-test")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "text-embedding-v4")
        self.assertEqual(body["input"], ["alpha", "beta"])
        self.assertEqual(body["dimensions"], 2)

    def test_rerank_uses_embeddings_cosine_similarity(self) -> None:
        calls = []
        responses = [
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.9, 0.1]},
                    {"index": 2, "embedding": [0.0, 1.0]},
                ]
            }
        ]

        def fake_urlopen(request, timeout: float):
            calls.append(request)
            return _FakeHTTPResponse(responses.pop(0))

        backend = AliyunEmbeddingBackend(
            "qwen",
            {
                "api_base_url": "https://dashscope.example.com/compatible-mode/v1",
                "api_key": "sk-test",
                "batch_size": 10,
            },
            urlopen=fake_urlopen,
        )

        results = backend.rerank("query", ["close", "far"], 1)

        self.assertEqual(len(calls), 1)
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], 0.9)


if __name__ == "__main__":
    unittest.main()
