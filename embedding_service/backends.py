"""
Backend factory for embedding and reranking models.
Supports local Qwen/BGE model families via sentence-transformers and cloud
embedding APIs that expose an OpenAI-compatible /embeddings endpoint, including
Alibaba Cloud Model Studio and SiliconFlow.

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

SILICONFLOW_EMBEDDING_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_EMBEDDING_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SILICONFLOW_EMBEDDING_DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"

API_EMBEDDING_DEFAULT_BATCH_SIZE = 10
KNOWN_PROVIDER_API_KEY_ENVS = {
    ALIYUN_EMBEDDING_DEFAULT_API_KEY_ENV,
    SILICONFLOW_EMBEDDING_DEFAULT_API_KEY_ENV,
}


class EmbeddingBackend(Protocol):
    """Protocol defining the embedding/rerank backend interface."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query. Returns [(index, score), ...]."""
        ...


class SentenceTransformerBackend:
    """Backend using sentence-transformers for both embedding and reranking."""

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
    name = str(raw_name or "").strip()
    if not name or name.startswith((".", "/")):
        return default_name
    alias = aliases.get(name.lower())
    if alias:
        return alias
    if name in MODEL_MAP:
        return default_name
    return name


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
