from dataclasses import dataclass
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


class WhisperBackend:
    """OpenAI Whisper via transformers (AutoModelForSpeechSeq2Seq)."""

    def __init__(self, model_config: dict[str, Any],
                 transcribe_config: dict[str, Any] | None = None,
                 vad_model: Any = None,
                 vad_get_speech_ts: Any = None) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_name = model_config["name"]
        device = model_config.get("device", "auto")
        torch_dtype_str = model_config.get("torch_dtype", "auto")

        if torch_dtype_str in ("", "auto"):
            self._torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype_str == "float16":
            self._torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            self._torch_dtype = torch.bfloat16
        else:
            self._torch_dtype = torch.float32

        if device in ("", "auto"):
            device_map = "auto" if torch.cuda.is_available() else None
        elif device == "cuda":
            device_map = "auto"
        else:
            device_map = None

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self._torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        self._vad_model = vad_model
        self._vad_get_speech_ts = vad_get_speech_ts

        vad_cfg = (transcribe_config or {}).get("vad") or {}
        self._vad_threshold = float(vad_cfg.get("threshold", 0.3))
        self._vad_min_speech_ms = int(vad_cfg.get("min_speech_duration_ms", 100))
        self._vad_min_silence_ms = int(vad_cfg.get("min_silence_duration_ms", 500))
        self._vad_speech_pad_ms = int(vad_cfg.get("speech_pad_ms", 600))

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
        import torch

        audio_array = self._load_audio(audio, sample_rate)

        if vad_filter and self._vad_model is not None and self._vad_get_speech_ts is not None:
            audio_array = self._apply_vad(audio_array, sample_rate)

        duration = float(audio_array.size) / float(sample_rate)

        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features

        device = next(self.model.parameters()).device
        if self._torch_dtype in (torch.float16, torch.bfloat16):
            input_features = input_features.to(device=device, dtype=self._torch_dtype)
        else:
            input_features = input_features.to(device=device)

        gen_kwargs: dict[str, Any] = {"return_timestamps": True}
        if beam_size > 1:
            gen_kwargs["num_beams"] = beam_size

        if language and language.lower() != "auto":
            gen_kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

        if initial_prompt:
            prompt_ids = self.processor.tokenizer.encode(initial_prompt, add_special_tokens=False)
            gen_kwargs["prompt_ids"] = prompt_ids[:224]

        with torch.no_grad():
            generated_ids = self.model.generate(input_features, **gen_kwargs)

        segments = self._decode_segments(generated_ids[0], duration)
        text = "".join(seg.text for seg in segments)

        return TranscriptionResult(
            text=text,
            language=language or "auto",
            language_probability=1.0,
            duration=duration,
            segments=segments,
        )

    def _load_audio(self, audio: str | np.ndarray, sample_rate: int) -> np.ndarray:
        if isinstance(audio, str):
            import librosa

            audio_array, _sr = librosa.load(audio, sr=sample_rate, mono=True)
            return audio_array.astype(np.float32)
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def _apply_vad(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        audio_tensor = torch.from_numpy(audio_array)
        speech_ts = self._vad_get_speech_ts(
            audio_tensor,
            self._vad_model,
            sampling_rate=sample_rate,
            threshold=self._vad_threshold,
            min_speech_duration_ms=self._vad_min_speech_ms,
            min_silence_duration_ms=self._vad_min_silence_ms,
            speech_pad_ms=self._vad_speech_pad_ms,
            return_seconds=False,
        )
        if not speech_ts:
            return np.array([], dtype=np.float32)
        segments = [audio_array[ts["start"]:ts["end"]] for ts in speech_ts]
        return np.concatenate(segments) if len(segments) > 1 else segments[0]

    def _decode_segments(self, generated_ids: Any, duration: float) -> list[TranscriptionSegment]:
        try:
            decoded = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=False, decode_with_timestamps=True
            )
            if isinstance(decoded, list):
                return [
                    TranscriptionSegment(
                        start=float(item["timestamp"][0]),
                        end=float(item["timestamp"][1]),
                        text=str(item["text"]).strip(),
                    )
                    for item in decoded
                    if str(item.get("text", "")).strip()
                ]
        except Exception:
            pass

        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return [TranscriptionSegment(start=0.0, end=duration, text=text)] if text else []


class FasterWhisperBackend:
    """CTranslate2-based faster-whisper backend."""

    def __init__(self, model_config: dict[str, Any],
                 transcribe_config: dict[str, Any] | None = None) -> None:
        from faster_whisper import WhisperModel

        kwargs: dict[str, Any] = {
            "model_size_or_path": model_config["name"],
            "device": model_config.get("device", "auto"),
            "compute_type": model_config.get("compute_type", "default"),
        }
        cpu_threads = model_config.get("cpu_threads", 0)
        if cpu_threads > 0:
            kwargs["cpu_threads"] = cpu_threads
        workers = model_config.get("workers", 0)
        if workers > 0:
            kwargs["num_workers"] = workers

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
            TranscriptionSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_iter
        ]
        text = "\n".join(s.text for s in segments if s.text)
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )


def create_asr_backend(model_config: dict[str, Any],
                       transcribe_config: dict[str, Any] | None = None,
                       vad_model: Any = None,
                       vad_get_speech_ts: Any = None) -> ASRBackend:
    provider = str(model_config.get("provider") or "whisper").strip().lower()
    if provider in {"whisper"}:
        return WhisperBackend(model_config, transcribe_config, vad_model, vad_get_speech_ts)
    if provider in {"faster-whisper", "faster_whisper"}:
        return FasterWhisperBackend(model_config, transcribe_config)
    raise ValueError(f"Unsupported ASR provider: {provider}")
