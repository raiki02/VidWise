"""
Backend factory for embedding and reranking models.
Supports Qwen and BGE model families via sentence-transformers.

Models are loaded via sentence-transformers which handles download/cache
automatically. Set HF_ENDPOINT env var to use a mirror:
    export HF_ENDPOINT=https://hf-mirror.com

If a local path is provided (name starting with ./ or /), it is loaded directly.
Otherwise the name is treated as a HuggingFace model ID.
"""

import os
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


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


# Maps shortcut names to HF model IDs (or local paths).
# sentence-transformers will auto-download and cache HF model IDs.
# Set HF_ENDPOINT env var for mirrors (e.g. https://hf-mirror.com).
MODEL_MAP = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "bge": "BAAI/bge-m3",
    "bge-large": "BAAI/bge-large-zh-v1.5",
}


def create_backend(name: str, device: str = "auto") -> EmbeddingBackend:
    """
    Create an embedding backend.

    - Shortcut names (qwen, bge, bge-large) are resolved to HF model IDs.
    - Paths starting with ./ or / are treated as local directories.
    - Any other string is used directly as a HF model ID.

    Usage:
        export HF_ENDPOINT=https://hf-mirror.com  # if behind GFW
        create_backend("qwen")                     # loads Qwen3-Embedding from HF
        create_backend("./models/bge-m3")           # loads local model
    """
    model_name = MODEL_MAP.get(name, name)
    return SentenceTransformerBackend(model_name, device)
