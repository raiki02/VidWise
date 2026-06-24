"""
Embedding and Rerank Service
FastAPI service providing /embed and /rerank endpoints.
Supports Qwen and BGE model backends.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure the embedding_service directory is on the path for local imports.
_svc_dir = os.path.dirname(os.path.abspath(__file__))
if _svc_dir not in sys.path:
    sys.path.insert(0, _svc_dir)

from backends import create_backend
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service", version="1.0.0")

# Configuration
# EMBEDDING_MODEL: shortcut name (qwen, bge, bge-large) or path to local model dir.
# Models must be downloaded first with huggingface-cli:
#   huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/qwen3-embedding
#   huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3
MODEL_PROVIDER = os.getenv("EMBEDDING_MODEL", "qwen")
MODEL_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

backend = None


def get_backend():
    global backend
    if backend is None:
        backend = create_backend(MODEL_PROVIDER, MODEL_DEVICE)
    return backend


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str = "qwen"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str = "qwen"
    top_k: int = 3


class RerankResult(BaseModel):
    index: int
    score: float
    text: str


class RerankResponse(BaseModel):
    results: list[RerankResult]


@app.get("/health")
def health():
    return {"status": "ok", "provider": MODEL_PROVIDER}


@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        be = get_backend()
        embeddings = be.embed(req.texts)
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
def rerank(req: RerankRequest):
    try:
        be = get_backend()
        top_k = min(req.top_k, len(req.documents))
        results = be.rerank(req.query, req.documents, top_k)
        return RerankResponse(
            results=[
                RerankResult(index=idx, score=score, text=req.documents[idx])
                for idx, score in results
            ]
        )
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
