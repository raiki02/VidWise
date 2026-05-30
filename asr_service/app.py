import logging
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import yaml


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.yaml"))
DEFAULT_ASR_CONFIG: dict[str, Any] = {
    "model": {
        "name": "small",
        "device": "auto",
        "compute_type": "default",
        "cpu_threads": 0,
        "workers": 1,
    },
    "transcribe": {
        "beam_size": 5,
        "vad_filter": True,
        "initial_prompt": "",
    },
}

app = FastAPI(title="video-extractor-asr")
model: WhisperModel | None = None
asr_config: dict[str, Any] = DEFAULT_ASR_CONFIG


def _configure_logging() -> None:
    # Uvicorn config may override log formatting; this ensures our app logger exists and emits.
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


_configure_logging()
logger = logging.getLogger("video_extractor.asr")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    status_code: int | None = None
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        # Ensure unexpected exceptions are visible in logs with a full traceback.
        logger.exception("unhandled exception method=%s path=%s", request.method, request.url.path)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Use INFO to make sure it shows up even when uvicorn access logs are off.
        if status_code is None:
            logger.info(
                "request completed method=%s path=%s status=%s elapsed_ms=%.2f",
                request.method,
                request.url.path,
                "-",
                elapsed_ms,
            )
        else:
            logger.info(
                "request completed method=%s path=%s status=%d elapsed_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )
    return response


@app.exception_handler(Exception)
async def log_exception_handler(request, exc: Exception):
    # This handler will log exceptions that are converted to 500 responses.
    # Returning HTTPException in endpoints is handled by FastAPI separately.
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
    # FastAPI/Starlette won't log HTTPException stack traces by default.
    # For 5xx, log server-side details to aid debugging.
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
    global asr_config, model

    try:
        asr_config = load_asr_config()
        model_config = asr_config["model"]

        kwargs: dict[str, Any] = {
            "model_size_or_path": model_config["name"],
            "device": model_config["device"],
            "compute_type": model_config["compute_type"],
        }
        if model_config["cpu_threads"] > 0:
            kwargs["cpu_threads"] = model_config["cpu_threads"]
        if model_config["workers"] > 0:
            kwargs["num_workers"] = model_config["workers"]

        logger.info("loading ASR model model=%s device=%s", model_config["name"], model_config["device"])
        model = WhisperModel(**kwargs)
        logger.info("ASR model loaded")
    except Exception:
        logger.exception("failed to load ASR model")
        raise


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("zh"),
    beam_size: int = Form(5),
    vad_filter: bool | None = Form(None),
    initial_prompt: str = Form(""),
) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="ASR model is not loaded")

    transcribe_config = asr_config["transcribe"]
    if beam_size <= 0:
        beam_size = transcribe_config["beam_size"]
    if vad_filter is None:
        vad_filter = transcribe_config["vad_filter"]
    if not initial_prompt:
        initial_prompt = transcribe_config["initial_prompt"]

    suffix = Path(file.filename or "audio").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)

    try:
        segments_iter, info = model.transcribe(
            tmp_path,
            language=language or None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt or None,
        )
        segments = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
            for segment in segments_iter
        ]
    except Exception as exc:
        logger.exception(
            "transcribe failed filename=%s suffix=%s language=%s beam_size=%s vad_filter=%s",
            file.filename,
            suffix,
            language,
            beam_size,
            vad_filter,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    text = "\n".join(segment["text"] for segment in segments if segment["text"])
    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments,
    }


def load_asr_config() -> dict[str, Any]:
    config = merge_dict(DEFAULT_ASR_CONFIG, {})
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = merge_dict(config, data.get("asr") or {})

    model_config = config["model"]
    transcribe_config = config["transcribe"]

    model_config["name"] = os.getenv("ASR_MODEL", model_config["name"])
    model_config["device"] = os.getenv("ASR_DEVICE", model_config["device"])
    model_config["compute_type"] = os.getenv("ASR_COMPUTE_TYPE", model_config["compute_type"])
    model_config["cpu_threads"] = int(os.getenv("ASR_CPU_THREADS", model_config["cpu_threads"]))
    model_config["workers"] = int(os.getenv("ASR_WORKERS", model_config["workers"]))

    transcribe_config["beam_size"] = int(transcribe_config["beam_size"])
    transcribe_config["vad_filter"] = bool(transcribe_config["vad_filter"])
    transcribe_config["initial_prompt"] = transcribe_config["initial_prompt"] or ""

    return config


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
