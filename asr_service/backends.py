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
    def __init__(self, model_config: dict[str, Any],
                 transcribe_config: dict[str, Any] | None = None) -> None:
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
        self.vad_parameters = (transcribe_config or {}).get("vad") or None

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
        transcribe_kwargs: dict[str, Any] = {
            "language": language or None,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "initial_prompt": initial_prompt or None,
            "condition_on_previous_text": False,
            "temperature": 0.0,
        }
        if vad_filter and self.vad_parameters:
            transcribe_kwargs["vad_parameters"] = self.vad_parameters

        segments_iter, info = self.model.transcribe(audio, **transcribe_kwargs)
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


class UltravoxBackend:
    def __init__(self, model_config: dict[str, Any]) -> None:
        import transformers

        self.model_name = model_config["name"]
        device = _ultravox_device(model_config["device"])
        self.pipe = transformers.pipeline(
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
        import librosa

        if isinstance(audio, str):
            audio_array, _sr = librosa.load(audio, sr=16000)
        else:
            audio_array = audio.astype(np.float32)
            # Ensure 1D
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=0)

        duration = _audio_duration(audio, sample_rate)

        prompt = _build_ultravox_prompt(language, initial_prompt)
        turns = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        generate_kwargs: dict[str, Any] = {"max_new_tokens": 256}
        if beam_size > 1:
            generate_kwargs["num_beams"] = beam_size

        result = self.pipe(
            {"audio": audio_array, "turns": turns, "sampling_rate": 16000},
            **generate_kwargs,
        )

        text = _extract_ultravox_text(result)
        return TranscriptionResult(
            text=text,
            language=language or "auto",
            language_probability=0.0,
            duration=duration,
            segments=[TranscriptionSegment(start=0.0, end=duration, text=text)] if text else [],
        )


def create_asr_backend(model_config: dict[str, Any],
                       transcribe_config: dict[str, Any] | None = None) -> ASRBackend:
    provider = str(model_config.get("provider") or "faster-whisper").strip().lower()
    if provider in {"faster-whisper", "faster_whisper", "whisper"}:
        return FasterWhisperBackend(model_config, transcribe_config)
    if provider in {"sensevoice", "sensevoice-small", "funasr"}:
        return SenseVoiceBackend(model_config)
    if provider in {"ultravox"}:
        return UltravoxBackend(model_config)
    raise ValueError("asr.model.provider must be one of: faster-whisper, sensevoice, ultravox")


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


def _ultravox_device(device: str) -> int | str:
    if device in {"", "auto"}:
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    if device == "cuda":
        return 0
    if device == "cpu":
        return -1
    try:
        return int(device)
    except ValueError:
        return -1


def _build_ultravox_prompt(language: str, initial_prompt: str) -> str:
    if initial_prompt:
        return f"{initial_prompt}<|audio|>"
    lang_name = _ultravox_language_name(language)
    if lang_name:
        return f"Transcribe the following audio in {lang_name}:<|audio|>"
    return "Transcribe this audio:<|audio|>"


def _ultravox_language_name(language: str) -> str | None:
    mapping = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "yue": "Cantonese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
        "pt": "Portuguese",
        "it": "Italian",
        "vi": "Vietnamese",
        "th": "Thai",
        "tr": "Turkish",
    }
    normalized = (language or "").strip().lower()
    if normalized in mapping:
        return mapping[normalized]
    for code, name in mapping.items():
        if normalized.startswith(code):
            return name
    return None


def _extract_ultravox_text(result: Any) -> str:
    if isinstance(result, list):
        parts = [_extract_ultravox_text(item) for item in result]
        return "\n".join(part for part in parts if part)
    if isinstance(result, dict):
        return str(result.get("generated_text") or result.get("text") or "").strip()
    return str(result or "").strip()
