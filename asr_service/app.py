import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel


MODEL_NAME = os.getenv("ASR_MODEL", "small")
DEVICE = os.getenv("ASR_DEVICE", "auto")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "default")
CPU_THREADS = int(os.getenv("ASR_CPU_THREADS", "0"))
WORKERS = int(os.getenv("ASR_WORKERS", "1"))

app = FastAPI(title="video-extractor-asr")
model: WhisperModel | None = None


@app.on_event("startup")
def load_model() -> None:
    global model

    kwargs: dict[str, Any] = {
        "model_size_or_path": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }
    if CPU_THREADS > 0:
        kwargs["cpu_threads"] = CPU_THREADS
    if WORKERS > 0:
        kwargs["num_workers"] = WORKERS

    model = WhisperModel(**kwargs)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("zh"),
    beam_size: int = Form(5),
) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="ASR model is not loaded")

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
            vad_filter=True,
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
