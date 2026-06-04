import os
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[TranscriptionSegment]


class ASRBackend(Protocol):
    def transcribe(
        self,
        audio: str | np.ndarray,
        *,
        language: str,
        beam_size: int,
        vad_filter: bool,
        initial_prompt: str,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        ...


class FasterWhisperBackend:
    def __init__(self, model_config: dict[str, Any]) -> None:
        from faster_whisper import WhisperModel

        kwargs: dict[str, Any] = {
            "model_size_or_path": model_config["name"],
            "device": model_config["device"],
            "compute_type": model_config["compute_type"],
        }
        if model_config["cpu_threads"] > 0:
            kwargs["cpu_threads"] = model_config["cpu_threads"]
        if model_config["workers"] > 0:
            kwargs["num_workers"] = model_config["workers"]

        self.model = WhisperModel(**kwargs)

    def transcribe(
        self,
        audio: str | np.ndarray,
        *,
        language: str,
        beam_size: int,
        vad_filter: bool,
        initial_prompt: str,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        segments_iter, info = self.model.transcribe(
            audio,
            language=language or None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt or None,
        )
        segments = [
            TranscriptionSegment(start=segment.start, end=segment.end, text=segment.text.strip())
            for segment in segments_iter
        ]
        text = "\n".join(segment.text for segment in segments if segment.text)
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )


class SenseVoiceBackend:
    def __init__(self, model_config: dict[str, Any]) -> None:
        from funasr import AutoModel

        self.model_name = model_config["name"]
        device = _sensevoice_device(model_config["device"])
        self.model = AutoModel(
            model=self.model_name,
            trust_remote_code=True,
            device=device,
        )

    def transcribe(
        self,
        audio: str | np.ndarray,
        *,
        language: str,
        beam_size: int,
        vad_filter: bool,
        initial_prompt: str,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        audio_path: str | None = None
        remove_audio = False
        if isinstance(audio, np.ndarray):
            audio_path = _write_pcm_wav(audio, sample_rate)
            remove_audio = True
        else:
            audio_path = audio

        try:
            result = self.model.generate(
                input=audio_path,
                cache={},
                language=_sensevoice_language(language),
                use_itn=True,
                batch_size_s=60,
                merge_vad=vad_filter,
                merge_length_s=15,
            )
        finally:
            if remove_audio and audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        text = _extract_sensevoice_text(result)
        duration = _audio_duration(audio, sample_rate)
        return TranscriptionResult(
            text=text,
            language=language or "auto",
            language_probability=0.0,
            duration=duration,
            segments=[TranscriptionSegment(start=0.0, end=duration, text=text)] if text else [],
        )


def create_asr_backend(model_config: dict[str, Any]) -> ASRBackend:
    provider = str(model_config.get("provider") or "faster-whisper").strip().lower()
    if provider in {"faster-whisper", "faster_whisper", "whisper"}:
        return FasterWhisperBackend(model_config)
    if provider in {"sensevoice", "sensevoice-small", "funasr"}:
        return SenseVoiceBackend(model_config)
    raise ValueError("asr.model.provider must be one of: faster-whisper, sensevoice")


def _sensevoice_device(device: str) -> str:
    if device in {"", "auto"}:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def _sensevoice_language(language: str) -> str:
    normalized = (language or "auto").lower()
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("ja"):
        return "ja"
    if normalized.startswith("ko"):
        return "ko"
    if normalized.startswith("yue"):
        return "yue"
    return "auto"


def _extract_sensevoice_text(result: Any) -> str:
    if isinstance(result, list):
        parts = [_extract_sensevoice_text(item) for item in result]
        return "\n".join(part for part in parts if part)
    if isinstance(result, dict):
        text = str(result.get("text") or "")
    else:
        text = str(result or "")

    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(text).strip()
    except Exception:
        return re.sub(r"<\|[^|]+?\|>", "", text).strip()


def _write_pcm_wav(audio: np.ndarray, sample_rate: int) -> str:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        path = tmp.name
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_i16.tobytes())
    return path


def _audio_duration(audio: str | np.ndarray, sample_rate: int) -> float:
    if isinstance(audio, np.ndarray):
        return float(audio.size) / float(sample_rate)
    if Path(audio).suffix.lower() == ".wav":
        try:
            with wave.open(audio, "rb") as wav:
                return float(wav.getnframes()) / float(wav.getframerate())
        except wave.Error:
            return 0.0
    return 0.0
