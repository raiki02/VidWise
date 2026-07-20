import json
import os
import unittest
from unittest.mock import patch

from embedding_service.backends import (
    AliyunEmbeddingBackend,
    AliyunRerankBackend,
    SiliconFlowEmbeddingBackend,
    SiliconFlowRerankBackend,
    create_backend,
    create_rerank_backend,
)


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


class AliyunRerankBackendTest(unittest.TestCase):
    def test_qwen3_rerank_posts_to_compatible_reranks_endpoint(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append((request, timeout))
            return _FakeHTTPResponse(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.42},
                    ]
                }
            )

        backend = AliyunRerankBackend(
            "qwen3-rerank",
            {
                "api_base_url": "https://dashscope.example.com/compatible-api/v1",
                "api_key": "ak-test",
                "api_timeout_seconds": 30,
                "instruction": "按事实相关性排序",
            },
            urlopen=fake_urlopen,
        )

        results = backend.rerank("query", ["doc-a", "doc-b"], 2)

        self.assertEqual(results, [(1, 0.91), (0, 0.42)])
        request, timeout = requests[0]
        self.assertEqual(timeout, 30)
        self.assertEqual(request.get_full_url(), "https://dashscope.example.com/compatible-api/v1/reranks")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer ak-test")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3-rerank")
        self.assertEqual(body["query"], "query")
        self.assertEqual(body["documents"], ["doc-a", "doc-b"])
        self.assertEqual(body["top_n"], 2)
        self.assertEqual(body["instruct"], "按事实相关性排序")

    def test_gte_rerank_posts_to_dashscope_text_rerank_endpoint(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append(request)
            return _FakeHTTPResponse(
                {
                    "output": {
                        "results": [
                            {"index": 0, "relevance_score": 0.81},
                            {"index": 1, "relevance_score": 0.73},
                        ]
                    }
                }
            )

        backend = AliyunRerankBackend(
            "gte",
            {
                "api_base_url": "https://dashscope.example.com/api/v1",
                "api_key": "ak-test",
                "return_documents": True,
            },
            urlopen=fake_urlopen,
        )

        results = backend.rerank("query", ["doc-a", "doc-b"], 1)

        self.assertEqual(results, [(0, 0.81)])
        request = requests[0]
        self.assertEqual(
            request.get_full_url(),
            "https://dashscope.example.com/api/v1/services/rerank/text-rerank/text-rerank",
        )
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "gte-rerank-v2")
        self.assertEqual(body["input"], {"query": "query", "documents": ["doc-a", "doc-b"]})
        self.assertEqual(body["parameters"], {"top_n": 1, "return_documents": True})


class SiliconFlowEmbeddingBackendTest(unittest.TestCase):
    def test_embed_posts_to_siliconflow_openai_compatible_endpoint(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append((request, timeout))
            return _FakeHTTPResponse({"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        backend = SiliconFlowEmbeddingBackend(
            "qwen",
            {
                "api_key": "sf-test",
                "dimensions": 1024,
                "batch_size": 10,
            },
            urlopen=fake_urlopen,
        )

        embeddings = backend.embed(["hello"])

        self.assertEqual(embeddings, [[0.1, 0.2]])
        request, timeout = requests[0]
        self.assertEqual(timeout, 120)
        self.assertEqual(request.get_full_url(), "https://api.siliconflow.cn/v1/embeddings")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer sf-test")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(body["input"], ["hello"])
        self.assertEqual(body["dimensions"], 1024)

    def test_provider_default_api_key_env_wins_over_other_known_provider_env(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append(request)
            return _FakeHTTPResponse({"data": [{"index": 0, "embedding": [1.0]}]})

        with patch.dict(
            os.environ,
            {"SILICONFLOW_API_KEY": "sf-env", "DASHSCOPE_API_KEY": "dashscope-env"},
            clear=True,
        ):
            backend = SiliconFlowEmbeddingBackend(
                "qwen",
                {"api_key_env": "DASHSCOPE_API_KEY"},
                urlopen=fake_urlopen,
            )
            backend.embed(["hello"])

        self.assertEqual(dict(requests[0].header_items())["Authorization"], "Bearer sf-env")

    def test_dimensions_are_omitted_for_siliconflow_bge_models(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append(request)
            return _FakeHTTPResponse({"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        with self.assertLogs("embedding_service.backends", level="WARNING") as logs:
            backend = SiliconFlowEmbeddingBackend(
                "BAAI/bge-m3",
                {
                    "api_key": "sf-test",
                    "dimensions": 1024,
                    "batch_size": 10,
                },
                urlopen=fake_urlopen,
            )

        embeddings = backend.embed(["hello"])

        self.assertIn("Ignoring embedding.dimensions=1024", logs.output[0])
        self.assertEqual(embeddings, [[0.1, 0.2]])
        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(body["model"], "BAAI/bge-m3")
        self.assertNotIn("dimensions", body)

    def test_factory_accepts_siliconflow_alias_provider(self) -> None:
        backend = create_backend("qwen3-8b", provider="silicon-flow", config={"api_key": "sf-test"})

        self.assertIsInstance(backend, SiliconFlowEmbeddingBackend)
        self.assertEqual(backend.model_name, "Qwen/Qwen3-Embedding-8B")


class SiliconFlowRerankBackendTest(unittest.TestCase):
    def test_rerank_posts_to_siliconflow_rerank_endpoint(self) -> None:
        requests = []

        def fake_urlopen(request, timeout: float):
            requests.append((request, timeout))
            return _FakeHTTPResponse(
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.12},
                        {"index": 1, "relevance_score": 0.98},
                    ]
                }
            )

        backend = SiliconFlowRerankBackend(
            "bge-reranker",
            {
                "api_base_url": "https://siliconflow.example.com/v1",
                "api_key": "sf-test",
                "api_timeout_seconds": 45,
                "instruction": "按视频转写语义相关性排序",
                "return_documents": True,
                "max_chunks_per_doc": 512,
                "overlap_tokens": 80,
            },
            urlopen=fake_urlopen,
        )

        results = backend.rerank("query", ["doc-a", "doc-b"], 2)

        self.assertEqual(results, [(1, 0.98), (0, 0.12)])
        request, timeout = requests[0]
        self.assertEqual(timeout, 45)
        self.assertEqual(request.get_full_url(), "https://siliconflow.example.com/v1/rerank")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer sf-test")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(body["query"], "query")
        self.assertEqual(body["documents"], ["doc-a", "doc-b"])
        self.assertEqual(body["top_n"], 2)
        self.assertEqual(body["instruction"], "按视频转写语义相关性排序")
        self.assertTrue(body["return_documents"])
        self.assertEqual(body["max_chunks_per_doc"], 512)
        self.assertEqual(body["overlap_tokens"], 80)

    def test_factory_accepts_siliconflow_rerank_provider(self) -> None:
        backend = create_rerank_backend("bge", provider="sf", config={"api_key": "sf-test"})

        self.assertIsInstance(backend, SiliconFlowRerankBackend)
        self.assertEqual(backend.model_name, "BAAI/bge-reranker-v2-m3")


if __name__ == "__main__":
    unittest.main()
