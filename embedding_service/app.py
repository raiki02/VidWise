"""
Embedding and Rerank Service
FastAPI service providing /embed and /rerank endpoints.
Supports local Qwen/BGE embedding models, dedicated local rerankers, and cloud
embedding/rerank APIs from Alibaba Cloud Model Studio and SiliconFlow.

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
  rerank.provider              RERANK_PROVIDER
  rerank.model                 RERANK_MODEL
  rerank.device                RERANK_DEVICE
  rerank.top_k                 RERANK_TOP_K
  rerank.api_base_url          RERANK_API_BASE_URL
  rerank.api_key               RERANK_API_KEY
  rerank.api_key_env           RERANK_API_KEY_ENV
  rerank.api_timeout_seconds   RERANK_API_TIMEOUT_SECONDS
  rerank.instruction           RERANK_INSTRUCTION
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

from backends import create_backend, create_rerank_backend
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


def _get_bool_env_or_cfg(env_name: str, cfg: dict, cfg_section: str, cfg_key: str, default: bool) -> bool:
    raw_env = os.getenv(env_name, "")
    if raw_env != "":
        return raw_env.strip().lower() in {"1", "true", "yes", "on"}
    raw_cfg = cfg.get(cfg_section, {}).get(cfg_key)
    if raw_cfg is not None:
        if isinstance(raw_cfg, bool):
            return raw_cfg
        return str(raw_cfg).strip().lower() in {"1", "true", "yes", "on"}
    return default


def _provider_group(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized in {"aliyun", "dashscope"}:
        return "aliyun"
    if normalized in {"siliconflow", "silicon-flow", "silicon_flow", "sf"}:
        return "siliconflow"
    return normalized


def _same_cloud_provider(left: str, right: str) -> bool:
    left_group = _provider_group(left)
    return left_group in {"aliyun", "siliconflow"} and left_group == _provider_group(right)


def _cloud_cfg_value(
    rerank_cfg: dict,
    embedding_cfg: dict,
    key: str,
    reuse_embedding: bool,
    default: object,
) -> object:
    value = rerank_cfg.get(key)
    if value not in (None, ""):
        return value
    if reuse_embedding:
        value = embedding_cfg.get(key)
        if value not in (None, ""):
            return value
    return default


_cfg = _load_config(CONFIG_PATH)
_embedding_cfg = _cfg.get("embedding", {}) or {}
_rerank_cfg = _cfg.get("rerank", {}) or {}

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
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", _rerank_cfg.get("provider", "local"))
RERANK_MODEL = os.getenv("RERANK_MODEL", _rerank_cfg.get("model", "BAAI/bge-reranker-v2-m3"))
RERANK_DEVICE = os.getenv("RERANK_DEVICE", _rerank_cfg.get("device", MODEL_DEVICE))
RERANK_TOP_K = _get_int_env_or_cfg("RERANK_TOP_K", _cfg, "rerank", "top_k", 3)
_reuse_embedding_api_config = _same_cloud_provider(RERANK_PROVIDER, MODEL_PROVIDER)
RERANK_API_BASE_URL = os.getenv(
    "RERANK_API_BASE_URL",
    str(_cloud_cfg_value(_rerank_cfg, _embedding_cfg, "api_base_url", _reuse_embedding_api_config, "")),
)
RERANK_API_KEY = os.getenv(
    "RERANK_API_KEY",
    str(_cloud_cfg_value(_rerank_cfg, _embedding_cfg, "api_key", _reuse_embedding_api_config, "")),
)
RERANK_API_KEY_ENV = os.getenv(
    "RERANK_API_KEY_ENV",
    str(_cloud_cfg_value(_rerank_cfg, _embedding_cfg, "api_key_env", _reuse_embedding_api_config, "")),
)
RERANK_API_TIMEOUT_SECONDS = float(os.getenv(
    "RERANK_API_TIMEOUT_SECONDS",
    _cloud_cfg_value(_rerank_cfg, _embedding_cfg, "api_timeout_seconds", _reuse_embedding_api_config, 120),
))
RERANK_INSTRUCTION = os.getenv("RERANK_INSTRUCTION", _rerank_cfg.get("instruction", ""))
RERANK_RETURN_DOCUMENTS = _get_bool_env_or_cfg("RERANK_RETURN_DOCUMENTS", _cfg, "rerank", "return_documents", False)
RERANK_MAX_CHUNKS_PER_DOC = _get_int_env_or_cfg("RERANK_MAX_CHUNKS_PER_DOC", _cfg, "rerank", "max_chunks_per_doc", 0)
RERANK_OVERLAP_TOKENS = _get_int_env_or_cfg("RERANK_OVERLAP_TOKENS", _cfg, "rerank", "overlap_tokens", 0)

EMBEDDING_PROVIDER_CONFIG = {
    "api_base_url": EMBEDDING_API_BASE_URL,
    "api_key": EMBEDDING_API_KEY,
    "api_key_env": EMBEDDING_API_KEY_ENV,
    "api_timeout_seconds": EMBEDDING_API_TIMEOUT_SECONDS,
    "dimensions": EMBEDDING_DIMENSIONS,
    "batch_size": EMBEDDING_BATCH_SIZE,
}

RERANK_PROVIDER_CONFIG = {
    "api_base_url": RERANK_API_BASE_URL,
    "api_key": RERANK_API_KEY,
    "api_key_env": RERANK_API_KEY_ENV,
    "api_timeout_seconds": RERANK_API_TIMEOUT_SECONDS,
    "instruction": RERANK_INSTRUCTION,
    "return_documents": RERANK_RETURN_DOCUMENTS,
    "max_chunks_per_doc": RERANK_MAX_CHUNKS_PER_DOC,
    "overlap_tokens": RERANK_OVERLAP_TOKENS,
}

logger.info(
    "Embedding service config: provider=%s model=%s device=%s rerank_provider=%s rerank_model=%s rerank_top_k=%d config_path=%s",
    MODEL_PROVIDER, MODEL_NAME, MODEL_DEVICE, RERANK_PROVIDER, RERANK_MODEL, RERANK_TOP_K, CONFIG_PATH,
)

embedding_backend = None
rerank_backend = None


def get_embedding_backend():
    global embedding_backend
    if embedding_backend is None:
        embedding_backend = create_backend(MODEL_NAME, MODEL_DEVICE, MODEL_PROVIDER, EMBEDDING_PROVIDER_CONFIG)
    return embedding_backend


def get_rerank_backend():
    global rerank_backend
    if rerank_backend is None:
        fallback_embedder = None
        if str(RERANK_PROVIDER).strip().lower() in {"embedding", "cosine", "legacy"}:
            fallback_embedder = get_embedding_backend()
        rerank_backend = create_rerank_backend(
            RERANK_MODEL,
            RERANK_DEVICE,
            RERANK_PROVIDER,
            RERANK_PROVIDER_CONFIG,
            fallback_embedder,
        )
    return rerank_backend


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
    return {
        "status": "ok",
        "embedding": {"provider": MODEL_PROVIDER, "model": MODEL_NAME, "device": MODEL_DEVICE},
        "rerank": {"provider": RERANK_PROVIDER, "model": RERANK_MODEL, "device": RERANK_DEVICE},
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        be = get_embedding_backend()
        embeddings = be.embed(req.texts)
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
def rerank(req: RerankRequest):
    try:
        be = get_rerank_backend()
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
