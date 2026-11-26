"""
Embedding Service
FastAPI service providing the /embed endpoint.
Supports local Qwen/BGE embedding models and cloud embedding APIs from
Alibaba Cloud Model Studio and SiliconFlow.

Configuration is read from config.yaml (same file used by the Go gateway),
with environment variables as overrides:

  config.yaml                    env var override
  embedding.provider           EMBEDDING_PROVIDER
  embedding.model              EMBEDDING_MODEL
  embedding.device             EMBEDDING_DEVICE
  embedding.api_base_url       EMBEDDING_API_BASE_URL
  embedding.api_key            EMBEDDING_API_KEY
  embedding.api_key_env        EMBEDDING_API_KEY_ENV
  embedding.api_timeout_seconds EMBEDDING_API_TIMEOUT_SECONDS
  embedding.dimensions         EMBEDDING_DIMENSIONS
  embedding.batch_size         EMBEDDING_BATCH_SIZE
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

from backends import EmbeddingBackend, create_backend
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service", version="1.0.0")

# --- Configuration -----------------------------------------------------------
# Priority: env var > config.yaml > hardcoded default

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


def _load_config(config_path: str) -> dict:
    """Load the embedding section from config.yaml."""
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
_embedding_cfg = _cfg.get("embedding", {}) or {}

MODEL_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    _embedding_cfg.get("provider", "local"),
)
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    _embedding_cfg.get("model", "./models/qwen3-embedding"),
)
MODEL_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    _embedding_cfg.get("device", "auto"),
)
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", _embedding_cfg.get("api_base_url", ""))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", _embedding_cfg.get("api_key", ""))
EMBEDDING_API_KEY_ENV = os.getenv("EMBEDDING_API_KEY_ENV", _embedding_cfg.get("api_key_env", ""))
EMBEDDING_API_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_API_TIMEOUT_SECONDS", _embedding_cfg.get("api_timeout_seconds", 120)))
EMBEDDING_DIMENSIONS = _get_int_env_or_cfg("EMBEDDING_DIMENSIONS", _cfg, "embedding", "dimensions", 0)
EMBEDDING_BATCH_SIZE = _get_int_env_or_cfg("EMBEDDING_BATCH_SIZE", _cfg, "embedding", "batch_size", 10)

EMBEDDING_PROVIDER_CONFIG = {
    "api_base_url": EMBEDDING_API_BASE_URL,
    "api_key": EMBEDDING_API_KEY,
    "api_key_env": EMBEDDING_API_KEY_ENV,
    "api_timeout_seconds": EMBEDDING_API_TIMEOUT_SECONDS,
    "dimensions": EMBEDDING_DIMENSIONS,
    "batch_size": EMBEDDING_BATCH_SIZE,
}

logger.info(
    "Embedding service config: provider=%s model=%s device=%s config_path=%s",
    MODEL_PROVIDER, MODEL_NAME, MODEL_DEVICE, CONFIG_PATH,
)

embedding_backend: EmbeddingBackend | None = None


@app.on_event("startup")
def load_embedding_backend() -> None:
    global embedding_backend
    try:
        embedding_backend = create_backend(MODEL_NAME, MODEL_DEVICE, MODEL_PROVIDER, EMBEDDING_PROVIDER_CONFIG)
        logger.info("Embedding backend loaded: provider=%s model=%s", MODEL_PROVIDER, MODEL_NAME)
    except Exception:
        logger.exception("failed to load embedding backend")
        raise


# --- Request / Response types ------------------------------------------------

class EmbedRequest(BaseModel):
    texts: list[str]
    model: str = "qwen"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


# --- Endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding": {"provider": MODEL_PROVIDER, "model": MODEL_NAME, "device": MODEL_DEVICE},
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    if embedding_backend is None:
        raise HTTPException(status_code=503, detail="embedding backend is not loaded")
    try:
        embeddings = embedding_backend.embed(req.texts)
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)