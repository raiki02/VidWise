"""
Backend factory for embedding and reranking models.
Supports local Qwen/BGE embedding models via sentence-transformers, dedicated
local CrossEncoder rerankers, and cloud APIs from Alibaba Cloud Model Studio
and SiliconFlow.

Models are loaded via sentence-transformers which handles download/cache
automatically. Set HF_ENDPOINT env var to use a mirror:
    export HF_ENDPOINT=https://hf-mirror.com

If a local path is provided (name starting with ./ or /), it is loaded directly.
Otherwise the name is treated as a HuggingFace model ID.
"""

import json
import math
import os
import logging
import urllib.error
import urllib.request
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ALIYUN_EMBEDDING_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_EMBEDDING_DEFAULT_MODEL = "text-embedding-v4"
ALIYUN_EMBEDDING_DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
ALIYUN_RERANK_COMPATIBLE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_RERANK_DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
ALIYUN_RERANK_DEFAULT_MODEL = "qwen3-rerank"

SILICONFLOW_EMBEDDING_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_EMBEDDING_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SILICONFLOW_EMBEDDING_DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"
SILICONFLOW_RERANK_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_RERANK_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

API_EMBEDDING_DEFAULT_BATCH_SIZE = 10
KNOWN_PROVIDER_API_KEY_ENVS = {
    ALIYUN_EMBEDDING_DEFAULT_API_KEY_ENV,
    SILICONFLOW_EMBEDDING_DEFAULT_API_KEY_ENV,
}


class EmbeddingBackend(Protocol):
    """Protocol defining the embedding backend interface."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...


class RerankBackend(Protocol):
    """Protocol defining the dedicated rerank backend interface."""

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query. Returns [(index, score), ...]."""
        ...


class SentenceTransformerBackend:
    """Embedding backend using sentence-transformers."""

    def __init__(self, model_name_or_path: str, device: str = "auto"):
        from sentence_transformers import SentenceTransformer
        from torch import mps
        # Resolve device: sentence-transformers doesn't accept "auto", convert to
        # "mps" on Apple Silicon or "cpu" otherwise.
        if device == "auto":
            device = "mps" if mps.is_available() else "cpu"
        logger.info(f"Loading model: {model_name_or_path} on device: {device}")
        self.model_name = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path, device=device)
        logger.info(f"Model loaded: {model_name_or_path}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        from sentence_transformers import util
        query_emb = self.model.encode(query, normalize_embeddings=True)
        doc_embs = self.model.encode(documents, normalize_embeddings=True)

        scores = util.cos_sim(query_emb, doc_embs)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in ranked[:top_k]]


class CrossEncoderRerankBackend:
    """Dedicated local reranker using sentence-transformers CrossEncoder."""

    def __init__(self, model_name_or_path: str, device: str = "auto"):
        from sentence_transformers import CrossEncoder
        from torch import mps

        if device == "auto":
            device = "mps" if mps.is_available() else "cpu"
        logger.info(f"Loading rerank model: {model_name_or_path} on device: {device}")
        self.model_name = model_name_or_path
        self.model = CrossEncoder(model_name_or_path, device=device)
        logger.info(f"Rerank model loaded: {model_name_or_path}")

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents or top_k <= 0:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [(int(idx), float(score)) for idx, score in ranked[:top_k]]


class CosineRerankBackend:
    """Backward-compatible reranker that scores candidates with embeddings."""

    def __init__(self, embedder: EmbeddingBackend):
        self.embedder = embedder

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents or top_k <= 0:
            return []
        vectors = self.embedder.embed([query] + documents)
        query_vector = vectors[0]
        doc_vectors = vectors[1:]
        ranked = sorted(
            enumerate(doc_vectors),
            key=lambda item: _cosine_similarity(query_vector, item[1]),
            reverse=True,
        )
        return [(idx, _cosine_similarity(query_vector, vec)) for idx, vec in ranked[:top_k]]


class OpenAICompatibleEmbeddingBackend:
    """Embedding backend for providers exposing OpenAI-compatible embeddings."""

    def __init__(
        self,
        model_name: str,
        *,
        provider_name: str,
        default_base_url: str,
        default_model_name: str,
        default_api_key_env: str,
        model_aliases: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ):
        config = config or {}
        self.provider_name = provider_name
        self.model_name = _api_embedding_model_name(model_name, default_model_name, model_aliases or {})
        self.api_base_url = str(
            config.get("api_base_url")
            or config.get("base_url")
            or default_base_url
        ).rstrip("/")
        self.api_key = _resolve_api_key(config, default_api_key_env)
        if not self.api_key:
            raise ValueError(
                "embedding.api_key, embedding.api_key_env, "
                f"or {default_api_key_env} is required for {provider_name} provider"
            )
        self.timeout = float(config.get("api_timeout_seconds") or 120)
        self.dimensions = int(config.get("dimensions") or 0)
        self.batch_size = max(1, int(config.get("batch_size") or API_EMBEDDING_DEFAULT_BATCH_SIZE))
        self.urlopen = urlopen

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload: dict[str, Any] = {
                "model": self.model_name,
                "input": batch,
                "encoding_format": "float",
            }
            dimensions = self._dimensions_for_payload()
            if dimensions > 0:
                payload["dimensions"] = dimensions

            response = self._post_json("/embeddings", payload)
            data = response.get("data") or []
            if len(data) != len(batch):
                raise RuntimeError(
                    f"{self.provider_name} embedding count mismatch: got {len(data)}, want {len(batch)}"
                )
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            embeddings.extend([list(map(float, item["embedding"])) for item in ordered])
        return embeddings

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents or top_k <= 0:
            return []

        vectors = self.embed([query] + documents)
        query_vector = vectors[0]
        doc_vectors = vectors[1:]
        ranked = sorted(
            enumerate(doc_vectors),
            key=lambda item: _cosine_similarity(query_vector, item[1]),
            reverse=True,
        )
        return [(idx, _cosine_similarity(query_vector, vec)) for idx, vec in ranked[:top_k]]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.api_base_url + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} embedding returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"call {self.provider_name} embedding failed: {exc.reason}") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"decode {self.provider_name} embedding response failed") from exc

    def _dimensions_for_payload(self) -> int:
        return self.dimensions


class AliyunEmbeddingBackend(OpenAICompatibleEmbeddingBackend):
    """Alibaba Cloud Model Studio embeddings via the OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ):
        super().__init__(
            model_name,
            provider_name="aliyun",
            default_base_url=ALIYUN_EMBEDDING_DEFAULT_BASE_URL,
            default_model_name=ALIYUN_EMBEDDING_DEFAULT_MODEL,
            default_api_key_env=ALIYUN_EMBEDDING_DEFAULT_API_KEY_ENV,
            config=config,
            urlopen=urlopen,
        )


SILICONFLOW_EMBEDDING_MODEL_ALIASES = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-vl": "Qwen/Qwen3-VL-Embedding-8B",
    "bge": "BAAI/bge-m3",
    "bge-m3": "BAAI/bge-m3",
    "bge-large": "BAAI/bge-large-zh-v1.5",
    "pro-bge-m3": "Pro/BAAI/bge-m3",
}


class SiliconFlowEmbeddingBackend(OpenAICompatibleEmbeddingBackend):
    """SiliconFlow embeddings via its OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ):
        super().__init__(
            model_name,
            provider_name="siliconflow",
            default_base_url=SILICONFLOW_EMBEDDING_DEFAULT_BASE_URL,
            default_model_name=SILICONFLOW_EMBEDDING_DEFAULT_MODEL,
            default_api_key_env=SILICONFLOW_EMBEDDING_DEFAULT_API_KEY_ENV,
            model_aliases=SILICONFLOW_EMBEDDING_MODEL_ALIASES,
            config=config,
            urlopen=urlopen,
        )
        if self.dimensions > 0 and not _siliconflow_model_supports_dimensions(self.model_name):
            logger.warning(
                "Ignoring embedding.dimensions=%d for SiliconFlow model %s; "
                "this parameter is only sent for Qwen3 embedding models",
                self.dimensions,
                self.model_name,
            )

    def _dimensions_for_payload(self) -> int:
        if not _siliconflow_model_supports_dimensions(self.model_name):
            return 0
        return self.dimensions


class HTTPRerankBackend:
    """Base class for provider rerank APIs."""

    def __init__(
        self,
        model_name: str,
        *,
        provider_name: str,
        default_base_url: str,
        default_model_name: str,
        default_api_key_env: str,
        model_aliases: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ):
        config = config or {}
        self.provider_name = provider_name
        self.model_name = _api_model_name(model_name, default_model_name, model_aliases or {})
        self.api_base_url = str(
            config.get("api_base_url")
            or config.get("base_url")
            or default_base_url
        ).rstrip("/")
        self.api_key = _resolve_api_key(config, default_api_key_env)
        if not self.api_key:
            raise ValueError(
                "rerank.api_key, rerank.api_key_env, "
                f"or {default_api_key_env} is required for {provider_name} rerank provider"
            )
        self.timeout = float(config.get("api_timeout_seconds") or 120)
        self.instruction = str(config.get("instruction") or "").strip()
        self.return_documents = bool(config.get("return_documents") or False)
        self.max_chunks_per_doc = int(config.get("max_chunks_per_doc") or 0)
        self.overlap_tokens = int(config.get("overlap_tokens") or 0)
        self.urlopen = urlopen

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} rerank returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"call {self.provider_name} rerank failed: {exc.reason}") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"decode {self.provider_name} rerank response failed") from exc

    def _endpoint(self, suffix: str) -> str:
        base = self.api_base_url.rstrip("/")
        if base.endswith(suffix):
            return base
        return base + suffix


SILICONFLOW_RERANK_MODEL_ALIASES = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "bge-reranker": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bce": "netease-youdao/bce-reranker-base_v1",
}


class SiliconFlowRerankBackend(HTTPRerankBackend):
    """Dedicated reranker using SiliconFlow's /rerank API."""

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ):
        super().__init__(
            model_name,
            provider_name="siliconflow",
            default_base_url=SILICONFLOW_RERANK_DEFAULT_BASE_URL,
            default_model_name=SILICONFLOW_RERANK_DEFAULT_MODEL,
            default_api_key_env=SILICONFLOW_EMBEDDING_DEFAULT_API_KEY_ENV,
            model_aliases=SILICONFLOW_RERANK_MODEL_ALIASES,
            config=config,
            urlopen=urlopen,
        )

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents or top_k <= 0:
            return []
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }
        if self.instruction:
            payload["instruction"] = self.instruction
        if self.return_documents:
            payload["return_documents"] = True
        if self.max_chunks_per_doc > 0:
            payload["max_chunks_per_doc"] = self.max_chunks_per_doc
        if self.overlap_tokens > 0:
            payload["overlap_tokens"] = self.overlap_tokens

        response = self._post_json(self._endpoint("/rerank"), payload)
        return _parse_rerank_results(response.get("results") or [], top_k)


ALIYUN_RERANK_MODEL_ALIASES = {
    "qwen": "qwen3-rerank",
    "qwen3": "qwen3-rerank",
    "qwen3-rerank": "qwen3-rerank",
    "gte": "gte-rerank-v2",
    "gte-rerank": "gte-rerank-v2",
    "gte-rerank-v2": "gte-rerank-v2",
}


class AliyunRerankBackend(HTTPRerankBackend):
    """Dedicated reranker using Alibaba Cloud Model Studio/DashScope APIs."""

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ):
        resolved_model = _api_model_name(model_name, ALIYUN_RERANK_DEFAULT_MODEL, ALIYUN_RERANK_MODEL_ALIASES)
        default_base_url = (
            ALIYUN_RERANK_COMPATIBLE_DEFAULT_BASE_URL
            if _aliyun_rerank_uses_compatible_api(resolved_model)
            else ALIYUN_RERANK_DASHSCOPE_DEFAULT_BASE_URL
        )
        super().__init__(
            model_name,
            provider_name="aliyun",
            default_base_url=default_base_url,
            default_model_name=ALIYUN_RERANK_DEFAULT_MODEL,
            default_api_key_env=ALIYUN_EMBEDDING_DEFAULT_API_KEY_ENV,
            model_aliases=ALIYUN_RERANK_MODEL_ALIASES,
            config=config,
            urlopen=urlopen,
        )

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents or top_k <= 0:
            return []
        if _aliyun_rerank_uses_compatible_api(self.model_name):
            response = self._rerank_compatible_api(query, documents, top_k)
        else:
            response = self._rerank_dashscope_api(query, documents, top_k)
        results = response.get("results")
        if results is None:
            results = (response.get("output") or {}).get("results") or []
        return _parse_rerank_results(results, top_k)

    def _rerank_compatible_api(self, query: str, documents: list[str], top_k: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }
        if self.instruction:
            payload["instruct"] = self.instruction
        return self._post_json(self._endpoint("/reranks"), payload)

    def _rerank_dashscope_api(self, query: str, documents: list[str], top_k: int) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "top_n": min(top_k, len(documents)),
            "return_documents": self.return_documents,
        }
        payload = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": parameters,
        }
        return self._post_json(self._endpoint("/services/rerank/text-rerank/text-rerank"), payload)


def _siliconflow_model_supports_dimensions(model_name: str) -> bool:
    return str(model_name).lower().startswith("qwen/qwen3-")


def _resolve_api_key(config: dict[str, Any], default_env: str) -> str:
    explicit = str(config.get("api_key") or "").strip()
    if explicit:
        return explicit
    env_name = str(config.get("api_key_env") or default_env).strip()
    if env_name in KNOWN_PROVIDER_API_KEY_ENVS and env_name != default_env:
        env_candidates = [default_env, env_name]
    else:
        env_candidates = [env_name]
        if default_env and default_env not in env_candidates:
            env_candidates.append(default_env)
    for candidate in env_candidates:
        if not candidate:
            continue
        value = os.getenv(candidate, "").strip()
        if value:
            return value
    return ""


def _api_embedding_model_name(raw_name: str, default_name: str, aliases: dict[str, str]) -> str:
    return _api_model_name(raw_name, default_name, aliases)


def _api_model_name(raw_name: str, default_name: str, aliases: dict[str, str]) -> str:
    name = str(raw_name or "").strip()
    if not name or name.startswith((".", "/")):
        return default_name
    alias = aliases.get(name.lower())
    if alias:
        return alias
    if name in MODEL_MAP:
        return default_name
    return name


def _parse_rerank_results(results: list[dict[str, Any]], top_k: int) -> list[tuple[int, float]]:
    parsed: list[tuple[int, float]] = []
    for item in results:
        if "index" not in item:
            continue
        try:
            idx = int(item["index"])
            score = float(item.get("relevance_score", item.get("score", 0.0)))
        except (TypeError, ValueError):
            continue
        parsed.append((idx, score))
    parsed.sort(key=lambda item: item[1], reverse=True)
    return parsed[:top_k]


def _aliyun_rerank_uses_compatible_api(model_name: str) -> bool:
    return str(model_name).strip().lower() == "qwen3-rerank"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# Maps shortcut names to HF model IDs (or local paths).
# sentence-transformers will auto-download and cache HF model IDs.
# Set HF_ENDPOINT env var for mirrors (e.g. https://hf-mirror.com).
MODEL_MAP = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "bge": "BAAI/bge-m3",
    "bge-large": "BAAI/bge-large-zh-v1.5",
}

RERANK_MODEL_MAP = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "bge-reranker": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "qwen3-reranker": "Qwen/Qwen3-Reranker-0.6B",
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
}


def create_backend(
    name: str,
    device: str = "auto",
    provider: str = "local",
    config: dict[str, Any] | None = None,
) -> EmbeddingBackend:
    """
    Create an embedding backend.

    - provider=local/sentence-transformers loads a local or HF model.
    - provider=aliyun/dashscope calls Alibaba Cloud Model Studio.
    - provider=siliconflow calls SiliconFlow.
    - Shortcut names (qwen, bge, bge-large) are resolved to HF model IDs.
    - Paths starting with ./ or / are treated as local directories.
    - Any other string is used directly as a HF model ID.

    Usage:
        export HF_ENDPOINT=https://hf-mirror.com  # if behind GFW
        create_backend("qwen")                     # loads Qwen3-Embedding from HF
        create_backend("./models/bge-m3")           # loads local model
    """
    provider = str(provider or "local").strip().lower()
    if provider in {"aliyun", "dashscope"}:
        return AliyunEmbeddingBackend(name, config)
    if provider in {"siliconflow", "silicon-flow", "silicon_flow", "sf"}:
        return SiliconFlowEmbeddingBackend(name, config)
    if provider not in {"local", "sentence-transformers", "sentence_transformers", "huggingface", "hf"}:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    model_name = MODEL_MAP.get(name, name)
    return SentenceTransformerBackend(model_name, device)


def create_rerank_backend(
    name: str,
    device: str = "auto",
    provider: str = "local",
    config: dict[str, Any] | None = None,
    embedding_backend: EmbeddingBackend | None = None,
) -> RerankBackend:
    """
    Create a rerank backend.

    - provider=local/cross-encoder loads a dedicated CrossEncoder reranker.
    - provider=aliyun/dashscope calls Alibaba Cloud's dedicated rerank API.
    - provider=siliconflow calls SiliconFlow's dedicated rerank API.
    - provider=embedding/cosine keeps the old embedding-similarity fallback.
    """
    provider = str(provider or "local").strip().lower()
    if provider in {"aliyun", "dashscope"}:
        return AliyunRerankBackend(name, config)
    if provider in {"siliconflow", "silicon-flow", "silicon_flow", "sf"}:
        return SiliconFlowRerankBackend(name, config)
    if provider in {"embedding", "cosine", "legacy"}:
        if embedding_backend is None:
            embedding_backend = create_backend(name, device=device, config=config)
        return CosineRerankBackend(embedding_backend)
    if provider not in {
        "local",
        "sentence-transformers",
        "sentence_transformers",
        "huggingface",
        "hf",
        "cross-encoder",
        "cross_encoder",
        "reranker",
    }:
        raise ValueError(f"Unsupported rerank provider: {provider}")

    model_name = RERANK_MODEL_MAP.get(str(name or "").strip().lower(), name)
    return CrossEncoderRerankBackend(model_name, device)
