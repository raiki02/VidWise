import logging
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import anyio
import inspect
import json

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import yaml

from asr_service.backends import ASRBackend, TranscriptionResult, create_asr_backend


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.yaml"))
DEFAULT_ASR_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "whisper",
        "name": "./models/whisper-small",
        "device": "auto",
        "torch_dtype": "auto",
    },
    "transcribe": {
        "beam_size": 5,
        "vad_filter": True,
        "initial_prompt": "",
        "vad": {
            "threshold": 0.3,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 600,
        },
    },
    "stream": {
        "enabled": True,
        "sample_rate": 16000,
        "chunk_size": 1600,
        "max_buffer_seconds": 30,
        "vad": {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 100,
            "max_speech_duration_s": 30,
        },
    },
}

app = FastAPI(title="video-extractor-asr")
model: ASRBackend | None = None
asr_config: dict[str, Any] = DEFAULT_ASR_CONFIG
vad_model: Any | None = None
vad_iterator_cls: Any | None = None


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
    global asr_config, model, vad_model, vad_iterator_cls

    try:
        asr_config = load_asr_config()
        model_config = asr_config["model"]

        stream_config = asr_config["stream"]
        vad_get_speech_ts = None
        if stream_config.get("enabled", True):
            logger.info("loading Silero VAD model")
            vad_model, vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            vad_iterator_cls = vad_utils[3]
            vad_get_speech_ts = vad_utils[2]
            logger.info("Silero VAD model loaded")

        logger.info(
            "loading ASR model provider=%s model=%s device=%s",
            model_config["provider"],
            model_config["name"],
            model_config["device"],
        )
        model = create_asr_backend(
            model_config,
            transcribe_config=asr_config["transcribe"],
            vad_model=vad_model,
            vad_get_speech_ts=vad_get_speech_ts,
        )
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
        result = model.transcribe(
            tmp_path,
            language=language or None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt or None,
        )
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

    return _result_payload(result)


@app.websocket("/stream")
async def stream_transcribe(websocket: WebSocket) -> None:
    if model is None:
        await websocket.close(code=1011)
        return
    if vad_model is None or vad_iterator_cls is None:
        await websocket.close(code=1011)
        return

    await websocket.accept()

    stream_config = asr_config["stream"]
    vad_config = stream_config["vad"].copy()
    sample_rate = int(stream_config["sample_rate"])
    max_buffer_samples = int(stream_config["max_buffer_seconds"] * sample_rate)
    beam_size = asr_config["transcribe"]["beam_size"]
    initial_prompt = asr_config["transcribe"]["initial_prompt"]
    language = "zh"

    vad_iterator = _create_vad_iterator(sample_rate, vad_config)
    stream_samples = 0
    speech_active = False
    speech_chunks: list[np.ndarray] = []
    speech_samples = 0
    segment_id = 0

    await websocket.send_json(
        {
            "type": "ready",
            "format": "pcm_s16le",
            "sample_rate": sample_rate,
            "chunk_size": int(stream_config["chunk_size"]),
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "invalid json message"})
                    continue

                event = payload.get("event")
                if event == "config":
                    language = payload.get("language") or language
                    beam_size = int(payload.get("beam_size") or beam_size)
                    initial_prompt = payload.get("initial_prompt") or initial_prompt
                    vad_override = payload.get("vad") or {}
                    if isinstance(vad_override, dict):
                        vad_config.update(vad_override)
                        vad_iterator = _create_vad_iterator(sample_rate, vad_config)
                    await websocket.send_json({"type": "config_ack"})
                    continue
                if event == "end":
                    if speech_active and speech_chunks:
                        await _finalize_stream_utterance(
                            websocket,
                            np.concatenate(speech_chunks),
                            language,
                            beam_size,
                            initial_prompt,
                            segment_id,
                        )
                        segment_id += 1
                    break
                continue

            if message.get("bytes"):
                audio = _decode_pcm(message["bytes"])
                if audio.size == 0:
                    continue

                chunk_start = stream_samples
                stream_samples += audio.size

                speech_event = vad_iterator(torch.from_numpy(audio), return_seconds=False)
                start_offset, end_offset = _event_offsets(speech_event, sample_rate, chunk_start, audio.size)

                if start_offset is not None:
                    speech_active = True
                    speech_chunks = []
                    speech_samples = 0
                    if start_offset < audio.size:
                        speech_chunks.append(audio[start_offset:])
                        speech_samples += audio.size - start_offset

                if speech_active and start_offset is None and end_offset is None:
                    speech_chunks.append(audio)
                    speech_samples += audio.size

                if speech_active and end_offset is not None:
                    if start_offset is not None:
                        if speech_chunks:
                            tail = speech_chunks[-1]
                            trim = max(0, min(tail.size, end_offset - start_offset))
                            speech_chunks[-1] = tail[:trim]
                    else:
                        speech_chunks.append(audio[:end_offset])
                        speech_samples += end_offset

                    await _finalize_stream_utterance(
                        websocket,
                        np.concatenate(speech_chunks) if speech_chunks else np.array([], dtype=np.float32),
                        language,
                        beam_size,
                        initial_prompt,
                        segment_id,
                    )
                    segment_id += 1
                    speech_active = False
                    speech_chunks = []
                    speech_samples = 0
                    vad_iterator.reset_states()

                if speech_active and speech_samples >= max_buffer_samples:
                    await _finalize_stream_utterance(
                        websocket,
                        np.concatenate(speech_chunks),
                        language,
                        beam_size,
                        initial_prompt,
                        segment_id,
                    )
                    segment_id += 1
                    speech_active = False
                    speech_chunks = []
                    speech_samples = 0
                    vad_iterator.reset_states()

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("stream transcribe failed")
        await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        if speech_active and speech_chunks:
            await _finalize_stream_utterance(
                websocket,
                np.concatenate(speech_chunks),
                language,
                beam_size,
                initial_prompt,
                segment_id,
            )


def _decode_pcm(data: bytes) -> np.ndarray:
    if not data:
        return np.array([], dtype=np.float32)
    audio_i16 = np.frombuffer(data, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0).copy()


def _event_offsets(event: Any, sample_rate: int, chunk_start: int, chunk_size: int) -> tuple[int | None, int | None]:
    if not event:
        return None, None

    start_offset = None
    end_offset = None

    if "start" in event:
        start_sample = _event_to_samples(event["start"], sample_rate)
        start_offset = max(0, start_sample - chunk_start)
        if start_offset > chunk_size:
            start_offset = None

    if "end" in event:
        end_sample = _event_to_samples(event["end"], sample_rate)
        end_offset = max(0, min(chunk_size, end_sample - chunk_start))

    return start_offset, end_offset


def _event_to_samples(value: Any, sample_rate: int) -> int:
    if isinstance(value, float):
        return int(value * sample_rate)
    return int(value)


def _create_vad_iterator(sample_rate: int, vad_config: dict[str, Any]) -> Any:
    if vad_iterator_cls is None or vad_model is None:
        raise RuntimeError("VAD model is not loaded")

    candidate_kwargs = {
        "sampling_rate": sample_rate,
        "threshold": float(vad_config.get("threshold", 0.5)),
        "min_speech_duration_ms": int(vad_config.get("min_speech_duration_ms", 250)),
        "min_silence_duration_ms": int(vad_config.get("min_silence_duration_ms", 200)),
        "speech_pad_ms": int(vad_config.get("speech_pad_ms", 100)),
        "max_speech_duration_s": float(vad_config.get("max_speech_duration_s", 30)),
    }
    signature = inspect.signature(vad_iterator_cls.__init__)
    valid_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }
    return vad_iterator_cls(vad_model, **valid_kwargs)


async def _finalize_stream_utterance(
    websocket: WebSocket,
    audio: np.ndarray,
    language: str,
    beam_size: int,
    initial_prompt: str,
    segment_id: int,
) -> None:
    if audio.size == 0:
        return

    try:
        segments, info, text = await _transcribe_audio(audio, language, beam_size, initial_prompt)
    except Exception as exc:
        logger.exception("stream transcribe chunk failed")
        await websocket.send_json({"type": "error", "detail": str(exc)})
        return

    await websocket.send_json(
        {
            "type": "final",
            "id": segment_id,
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
        }
    )


def _run_transcribe(audio: np.ndarray, language: str, beam_size: int, initial_prompt: str) -> tuple[list[dict[str, Any]], Any, str]:
    if model is None:
        raise RuntimeError("ASR model is not loaded")

    sample_rate = int(asr_config["stream"]["sample_rate"])
    result = model.transcribe(
        audio,
        language=language or None,
        beam_size=beam_size,
        vad_filter=False,
        initial_prompt=initial_prompt or None,
        sample_rate=sample_rate,
    )
    return _segments_payload(result), result, result.text


async def _transcribe_audio(
    audio: np.ndarray, language: str, beam_size: int, initial_prompt: str
) -> tuple[list[dict[str, Any]], Any, str]:
    return await anyio.to_thread.run_sync(_run_transcribe, audio, language, beam_size, initial_prompt)


def load_asr_config() -> dict[str, Any]:
    config = merge_dict(DEFAULT_ASR_CONFIG, {})
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = merge_dict(config, data.get("asr") or {})

    model_config = config["model"]
    transcribe_config = config["transcribe"]
    stream_config = config["stream"]
    vad_config = stream_config["vad"]

    model_config["provider"] = os.getenv("ASR_PROVIDER", model_config.get("provider", "whisper"))
    model_config["name"] = os.getenv("ASR_MODEL", model_config["name"])
    model_config["device"] = os.getenv("ASR_DEVICE", model_config["device"])
    model_config["torch_dtype"] = os.getenv("ASR_TORCH_DTYPE", model_config.get("torch_dtype", "auto"))

    transcribe_config["beam_size"] = int(transcribe_config["beam_size"])
    transcribe_config["vad_filter"] = bool(transcribe_config["vad_filter"])
    transcribe_config["initial_prompt"] = transcribe_config["initial_prompt"] or ""

    stream_config["enabled"] = bool(stream_config.get("enabled", True))
    stream_config["sample_rate"] = int(stream_config["sample_rate"])
    stream_config["chunk_size"] = int(stream_config["chunk_size"])
    stream_config["max_buffer_seconds"] = float(stream_config["max_buffer_seconds"])
    vad_config["threshold"] = float(vad_config["threshold"])
    vad_config["min_speech_duration_ms"] = int(vad_config["min_speech_duration_ms"])
    vad_config["min_silence_duration_ms"] = int(vad_config["min_silence_duration_ms"])
    vad_config["speech_pad_ms"] = int(vad_config["speech_pad_ms"])
    vad_config["max_speech_duration_s"] = float(vad_config["max_speech_duration_s"])

    return config


def _result_payload(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "segments": _segments_payload(result),
    }


def _segments_payload(result: TranscriptionResult) -> list[dict[str, Any]]:
    return [
        {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
        for segment in result.segments
    ]


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
