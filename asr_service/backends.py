import base64
import io
import json
import mimetypes
import os
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


DEFAULT_WHISPER_MAX_NEW_TOKENS = 448
DEFAULT_WHISPER_MAX_INITIAL_PROMPT_TOKENS = 224
MIN_WHISPER_MAX_NEW_TOKENS = 1
ALIYUN_ASR_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_ASR_DEFAULT_MODEL = "qwen3-asr-flash"
ALIYUN_ASR_DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
# Base64 grows payload size by roughly 4/3. Keep local-file requests safely under
# the 10 MB Qwen3-ASR-Flash input cap.
ALIYUN_ASR_DEFAULT_MAX_FILE_BYTES = 7_500_000


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
        self._max_new_tokens = _positive_int(
            (transcribe_config or {}).get("max_new_tokens"),
            DEFAULT_WHISPER_MAX_NEW_TOKENS,
        )
        self._max_initial_prompt_tokens = _positive_int(
            (transcribe_config or {}).get("max_initial_prompt_tokens"),
            DEFAULT_WHISPER_MAX_INITIAL_PROMPT_TOKENS,
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
        import torch

        audio_array = self._load_audio(audio, sample_rate)
        total_duration = float(audio_array.size) / float(sample_rate)

        if vad_filter and self._vad_model is not None and self._vad_get_speech_ts is not None:
            audio_array = self._apply_vad(audio_array, sample_rate)

        # Whisper-small positional encoding covers ~30 seconds.
        # For longer audio, chunk into overlapping windows to avoid silent truncation.
        max_chunk_samples = 25 * sample_rate  # 25s per chunk
        overlap_samples = 2 * sample_rate     # 2s overlap

        if audio_array.size <= max_chunk_samples:
            return self._transcribe_chunk(
                audio_array, sample_rate, language, beam_size, initial_prompt, total_duration,
            )

        import logging
        _log = logging.getLogger("vidwise.asr")
        _log.info(
            "whisper.long_audio_chunking samples=%d duration=%.1fs chunk_size=%ds",
            audio_array.size, total_duration, 25,
        )

        all_segments: list[TranscriptionSegment] = []
        offset = 0
        while offset < audio_array.size:
            chunk_end = min(offset + max_chunk_samples, audio_array.size)
            chunk = audio_array[offset:chunk_end]
            chunk_duration = float(chunk.size) / float(sample_rate)
            _log.info("whisper.chunk %d-%d/%d (%.1fs-%.1fs)", offset, chunk_end, audio_array.size, offset / sample_rate, chunk_end / sample_rate)
            result = self._transcribe_chunk(
                chunk, sample_rate, language, beam_size, initial_prompt, chunk_duration,
            )
            for seg in result.segments:
                seg.start += offset / sample_rate
                seg.end += offset / sample_rate
            all_segments.extend(result.segments)
            if chunk_end >= audio_array.size:
                break
            offset = chunk_end - overlap_samples

        text = "".join(seg.text for seg in all_segments)
        return TranscriptionResult(
            text=text,
            language=language or "auto",
            language_probability=1.0,
            duration=total_duration,
            segments=all_segments,
        )

    def _transcribe_chunk(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: str,
        beam_size: int,
        initial_prompt: str,
        duration: float,
    ) -> TranscriptionResult:
        import torch

        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features

        device = next(self.model.parameters()).device
        if self._torch_dtype in (torch.float16, torch.bfloat16):
            input_features = input_features.to(device=device, dtype=self._torch_dtype)
        else:
            input_features = input_features.to(device=device)

        gen_kwargs = _build_whisper_generation_kwargs(
            self.model,
            self.processor,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            requested_max_new_tokens=self._max_new_tokens,
            max_initial_prompt_tokens=self._max_initial_prompt_tokens,
        )

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


def _build_whisper_generation_kwargs(
    model: Any,
    processor: Any,
    *,
    language: str,
    beam_size: int,
    initial_prompt: str,
    requested_max_new_tokens: int = DEFAULT_WHISPER_MAX_NEW_TOKENS,
    max_initial_prompt_tokens: int = DEFAULT_WHISPER_MAX_INITIAL_PROMPT_TOKENS,
) -> dict[str, Any]:
    forced_decoder_ids = None
    gen_kwargs: dict[str, Any] = {
        "return_timestamps": True,
        # Suppress repeated n-grams to break hallucination loops.
        "no_repeat_ngram_size": 4,
    }
    if beam_size > 1:
        gen_kwargs["num_beams"] = beam_size

    if language and language.lower() != "auto":
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

    max_target_positions = _whisper_max_target_positions(model)
    decoder_prefix_tokens = _decoder_prefix_token_count(forced_decoder_ids)
    requested_max_new_tokens = _positive_int(
        requested_max_new_tokens,
        DEFAULT_WHISPER_MAX_NEW_TOKENS,
    )
    max_initial_prompt_tokens = _positive_int(
        max_initial_prompt_tokens,
        DEFAULT_WHISPER_MAX_INITIAL_PROMPT_TOKENS,
    )

    prompt_ids: list[int] = []
    if initial_prompt:
        encoded_prompt = processor.tokenizer.encode(initial_prompt, add_special_tokens=False)
        prompt_budget = max(0, max_target_positions - decoder_prefix_tokens - MIN_WHISPER_MAX_NEW_TOKENS)
        prompt_ids = encoded_prompt[:min(max_initial_prompt_tokens, prompt_budget)]
        if prompt_ids:
            gen_kwargs["prompt_ids"] = prompt_ids

    max_new_budget = max(
        MIN_WHISPER_MAX_NEW_TOKENS,
        max_target_positions - decoder_prefix_tokens - len(prompt_ids),
    )
    gen_kwargs["max_new_tokens"] = min(requested_max_new_tokens, max_new_budget)
    return gen_kwargs


def _whisper_max_target_positions(model: Any) -> int:
    for owner in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        for attr in ("max_target_positions", "max_length"):
            value = getattr(owner, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return DEFAULT_WHISPER_MAX_NEW_TOKENS


def _decoder_prefix_token_count(forced_decoder_ids: Any) -> int:
    # Whisper generation starts with <|startoftranscript|>; language/task forced
    # decoder ids are already present when transformers validates max length.
    return 1 + len(forced_decoder_ids or [])


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


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


class AliyunASRBackend:
    """Alibaba Cloud Model Studio Qwen-ASR via the OpenAI-compatible API."""

    def __init__(
        self,
        model_config: dict[str, Any],
        transcribe_config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
    ) -> None:
        self.model_name = _aliyun_model_name(model_config.get("name"))
        self.api_base_url = str(
            model_config.get("api_base_url")
            or model_config.get("base_url")
            or ALIYUN_ASR_DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = _resolve_api_key(model_config, ALIYUN_ASR_DEFAULT_API_KEY_ENV)
        if not self.api_key:
            raise ValueError("asr.model.api_key or asr.model.api_key_env is required for aliyun provider")

        self.timeout = float(model_config.get("api_timeout_seconds") or 300)
        self.max_file_bytes = int(model_config.get("max_file_bytes") or ALIYUN_ASR_DEFAULT_MAX_FILE_BYTES)
        self.enable_itn = bool((transcribe_config or {}).get("enable_itn", model_config.get("enable_itn", False)))
        self.urlopen = urlopen

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
        del beam_size, vad_filter

        audio_input, duration = self._audio_input(audio, sample_rate)
        messages: list[dict[str, Any]] = []
        if initial_prompt:
            messages.append({"role": "system", "content": initial_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_input},
                    }
                ],
            }
        )

        asr_options: dict[str, Any] = {"enable_itn": self.enable_itn}
        if language and language.lower() != "auto":
            asr_options["language"] = language

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "asr_options": asr_options,
        }
        response = self._post_json("/chat/completions", payload)
        text, detected_language = _parse_aliyun_asr_response(response, language)
        duration = float((response.get("usage") or {}).get("seconds") or duration)

        segments = [TranscriptionSegment(start=0.0, end=duration, text=text)] if text else []
        return TranscriptionResult(
            text=text,
            language=detected_language or language or "auto",
            language_probability=1.0,
            duration=duration,
            segments=segments,
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.api_base_url + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"aliyun asr returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"call aliyun asr failed: {exc.reason}") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("decode aliyun asr response failed") from exc

    def _audio_input(self, audio: str | np.ndarray, sample_rate: int) -> tuple[str, float]:
        if isinstance(audio, np.ndarray):
            return _ndarray_to_wav_data_url(audio, sample_rate), float(audio.size) / float(sample_rate)

        if audio.startswith(("http://", "https://", "oss://")):
            return audio, 0.0

        stat = os.stat(audio)
        if stat.st_size > self.max_file_bytes:
            raise ValueError(
                f"audio file is {stat.st_size} bytes, exceeds aliyun base64 limit "
                f"configured by asr.model.max_file_bytes={self.max_file_bytes}"
            )
        mime_type = mimetypes.guess_type(audio)[0] or "application/octet-stream"
        with open(audio, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}", _wav_duration(audio)


def _resolve_api_key(config: dict[str, Any], default_env: str) -> str:
    explicit = str(config.get("api_key") or "").strip()
    if explicit:
        return explicit
    env_name = str(config.get("api_key_env") or default_env).strip()
    if not env_name:
        return ""
    return os.getenv(env_name, "").strip()


def _aliyun_model_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    if not name or name.startswith((".", "/")):
        return ALIYUN_ASR_DEFAULT_MODEL
    return name


def _ndarray_to_wav_data_url(audio: np.ndarray, sample_rate: int) -> str:
    audio_i16 = (np.clip(audio.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        return float(frames) / float(rate) if rate else 0.0
    except (wave.Error, OSError, EOFError):
        return 0.0


def _parse_aliyun_asr_response(payload: dict[str, Any], requested_language: str) -> tuple[str, str]:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("aliyun asr response did not include choices")

    message = choices[0].get("message") or {}
    text = _message_content_text(message.get("content")).strip()
    annotations = message.get("annotations") or []
    detected_language = requested_language or "auto"
    for annotation in annotations:
        if isinstance(annotation, dict) and annotation.get("language"):
            detected_language = str(annotation["language"])
            break
    return text, detected_language


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
        return "".join(parts)
    return ""


def create_asr_backend(model_config: dict[str, Any],
                       transcribe_config: dict[str, Any] | None = None,
                       vad_model: Any = None,
                       vad_get_speech_ts: Any = None) -> ASRBackend:
    provider = str(model_config.get("provider") or "whisper").strip().lower()
    if provider in {"whisper"}:
        return WhisperBackend(model_config, transcribe_config, vad_model, vad_get_speech_ts)
    if provider in {"faster-whisper", "faster_whisper"}:
        return FasterWhisperBackend(model_config, transcribe_config)
    if provider in {"aliyun", "dashscope"}:
        return AliyunASRBackend(model_config, transcribe_config)
    raise ValueError(f"Unsupported ASR provider: {provider}")
