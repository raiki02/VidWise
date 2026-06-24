"""
Backend factory for embedding and reranking models.
Supports Qwen and BGE model families via sentence-transformers.
"""

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

    def __init__(self, model_name: str, device: str = "auto"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading model: {model_name} on device: {device}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded: {model_name}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        # Use sentence-transformers CrossEncoder-style reranking
        # or cosine similarity if cross-encoder not available
        from sentence_transformers import util
        query_emb = self.model.encode(query, normalize_embeddings=True)
        doc_embs = self.model.encode(documents, normalize_embeddings=True)

        scores = util.cos_sim(query_emb, doc_embs)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in ranked[:top_k]]


# Model name mappings
MODEL_MAP = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "bge": "BAAI/bge-m3",
    "bge-large": "BAAI/bge-large-zh-v1.5",
}


def create_backend(name: str, device: str = "auto") -> EmbeddingBackend:
    """
    Create an embedding backend by name.
    Supported: qwen, bge, bge-large, or a full HuggingFace model ID.
    """
    model_name = MODEL_MAP.get(name, name)
    return SentenceTransformerBackend(model_name, device)
