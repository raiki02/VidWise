import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


@app.on_event("startup")
def load_model() -> None:
    global asr_config, model

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

    model = WhisperModel(**kwargs)


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
