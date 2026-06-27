"""
Embedding and Rerank Service
FastAPI service providing /embed and /rerank endpoints.
Supports Qwen and BGE model backends.

Configuration is read from config.yaml (same file used by the Go gateway),
with environment variables as overrides:

  config.yaml                    env var override
  embedding.model              EMBEDDING_MODEL
  embedding.device             EMBEDDING_DEVICE
  rerank.top_k                 RERANK_TOP_K
"""

import os
import sys

import yaml
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

# --- Configuration -----------------------------------------------------------
# Priority: env var > config.yaml > hardcoded default

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


def _load_config(config_path: str) -> dict:
    """Load the embedding and rerank sections from config.yaml."""
    if not os.path.isfile(config_path):
        logger.warning("Config file %s not found, using defaults", config_path)
        return {}
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _get_int_env_or_cfg(env_name: str, cfg: dict, cfg_section: str, cfg_key: str, default: int) -> int:
    """Read an integer from env var, falling back to config.yaml, then the default.
    Guards against empty-string env vars and null YAML values."""
    raw_env = os.getenv(env_name, "")
    if raw_env != "":
        return int(raw_env)
    raw_cfg = cfg.get(cfg_section, {}).get(cfg_key)
    if raw_cfg is not None:
        return int(raw_cfg)
    return default


_cfg = _load_config(CONFIG_PATH)

MODEL_PROVIDER = os.getenv(
    "EMBEDDING_MODEL",
    _cfg.get("embedding", {}).get("model", "./models/qwen3-embedding"),
)
MODEL_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    _cfg.get("embedding", {}).get("device", "auto"),
)
RERANK_TOP_K = _get_int_env_or_cfg("RERANK_TOP_K", _cfg, "rerank", "top_k", 3)

logger.info(
    "Embedding service config: model=%s device=%s rerank_top_k=%d config_path=%s",
    MODEL_PROVIDER, MODEL_DEVICE, RERANK_TOP_K, CONFIG_PATH,
)

backend = None


def get_backend():
    global backend
    if backend is None:
        backend = create_backend(MODEL_PROVIDER, MODEL_DEVICE)
    return backend


# --- Request / Response types ------------------------------------------------

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


# --- Endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "provider": MODEL_PROVIDER, "device": MODEL_DEVICE}


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
        top_k = min(req.top_k or RERANK_TOP_K, len(req.documents))
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
