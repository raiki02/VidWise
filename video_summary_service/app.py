import logging
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .backends import create_backend, VideoSummaryBackend


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.yaml"))
DEFAULT_VIDEO_SUMMARY_CONFIG: dict[str, Any] = {
    "model": {
        "name": "./models/marlin",
        "provider": "huggingface",
        "device": "auto",
        "dtype": "bfloat16",
        "compile": False,
    },
    "summarize": {
        "max_new_tokens": 2048,
        "prompt": "",
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    },
}

app = FastAPI(title="video-extractor-video-summary")
model: VideoSummaryBackend | None = None
video_summary_config: dict[str, Any] = DEFAULT_VIDEO_SUMMARY_CONFIG
model_lock = threading.Lock()


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


_configure_logging()
logger = logging.getLogger("video_extractor.video_summary")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    status_code: int | None = None
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        logger.exception("unhandled exception method=%s path=%s", request.method, request.url.path)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request completed method=%s path=%s status=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            status_code if status_code is not None else "-",
            elapsed_ms,
        )
    return response


@app.exception_handler(Exception)
async def log_exception_handler(request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "500 internal error method=%s path=%s error=%r\n%s",
        request.method,
        request.url.path,
        exc,
        tb,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.exception_handler(HTTPException)
async def log_http_exception_handler(request, exc: HTTPException):
    if exc.status_code >= 500:
        logger.error(
            "HTTPException %d method=%s path=%s detail=%r",
            exc.status_code,
            request.method,
            request.url.path,
            exc.detail,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.on_event("startup")
def load_model() -> None:
    global model, video_summary_config

    try:
        video_summary_config = load_video_summary_config()
        model_config = video_summary_config["model"]

        logger.info(
            "loading video summary model model=%s provider=%s",
            model_config["name"],
            model_config.get("provider", "huggingface"),
        )
        model = create_backend(model_config)
        logger.info("video summary model loaded")
    except Exception:
        logger.exception("failed to load video summary model")
        raise


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/caption")
async def caption(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(0),
    prompt: str = Form(""),
    do_sample: bool | None = Form(None),
    temperature: float = Form(0),
    top_p: float = Form(0),
) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="video summary model is not loaded")

    summarize_config = video_summary_config["summarize"]
    if max_new_tokens <= 0:
        max_new_tokens = 64
    if not prompt:
        prompt = summarize_config["prompt"]
    if do_sample is None:
        do_sample = summarize_config["do_sample"]
    if temperature <= 0:
        temperature = summarize_config["temperature"]
    if top_p <= 0:
        top_p = summarize_config["top_p"]

    tmp_path = await _save_upload(file, "video", ".mp4")
    try:
        with model_lock:
            result = model.caption(
                tmp_path,
                max_new_tokens=max_new_tokens,
                prompt=prompt or None,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
    except Exception as exc:
        logger.exception("caption failed filename=%s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _remove_file(tmp_path)

    return {
        "caption": result.get("caption", ""),
        "scene": result.get("scene", ""),
        "events": result.get("events", []),
    }


@app.post("/find")
async def find_event(
    file: UploadFile = File(...),
    event: str = Form(...),
    max_new_tokens: int = Form(0),
    do_sample: bool | None = Form(None),
    temperature: float = Form(0),
    top_p: float = Form(0),
) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="video summary model is not loaded")
    event = event.strip()
    if not event:
        raise HTTPException(status_code=400, detail="event is required")

    summarize_config = video_summary_config["summarize"]
    if max_new_tokens <= 0:
        max_new_tokens = summarize_config["max_new_tokens"]
    if do_sample is None:
        do_sample = summarize_config["do_sample"]
    if temperature <= 0:
        temperature = summarize_config["temperature"]
    if top_p <= 0:
        top_p = summarize_config["top_p"]

    tmp_path = await _save_upload(file, "video", ".mp4")
    try:
        with model_lock:
            result = model.find(
                tmp_path,
                event=event,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
    except Exception as exc:
        logger.exception("find failed filename=%s event=%s", file.filename, event)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _remove_file(tmp_path)

    span = result.get("span")
    return {
        "raw": result.get("raw", ""),
        "span": list(span) if span is not None else None,
        "format_ok": bool(result.get("format_ok", False)),
    }


async def _save_upload(file: UploadFile, stem: str, default_suffix: str) -> str:
    suffix = Path(file.filename or stem).suffix or default_suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
    return tmp_path


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def load_video_summary_config() -> dict[str, Any]:
    config = merge_dict(DEFAULT_VIDEO_SUMMARY_CONFIG, {})
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = merge_dict(config, data.get("video_summary") or {})

    model_config = config["model"]
    summarize_config = config["summarize"]

    model_config["name"] = os.getenv("VIDEO_SUMMARY_MODEL", model_config["name"])
    model_config["provider"] = os.getenv("VIDEO_SUMMARY_PROVIDER", model_config.get("provider", "huggingface"))
    model_config["device"] = os.getenv("VIDEO_SUMMARY_DEVICE", model_config["device"])
    model_config["dtype"] = os.getenv("VIDEO_SUMMARY_DTYPE", model_config["dtype"])
    model_config["compile"] = _parse_bool(os.getenv("VIDEO_SUMMARY_COMPILE", model_config["compile"]))

    summarize_config["max_new_tokens"] = int(summarize_config["max_new_tokens"])
    summarize_config["prompt"] = summarize_config["prompt"] or ""
    summarize_config["do_sample"] = _parse_bool(summarize_config["do_sample"])
    summarize_config["temperature"] = float(summarize_config["temperature"])
    summarize_config["top_p"] = float(summarize_config["top_p"])

    return config


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {
        key: merge_dict(value, {}) if isinstance(value, dict) else value
        for key, value in base.items()
    }
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged
