import base64
import hashlib
import hmac
import io
import json
import mimetypes
import os
import secrets
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
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
XFYUN_ASR_DEFAULT_BASE_URL = "https://raasr.xfyun.cn/v2"
XFYUN_ASR_DEFAULT_APP_ID_ENV = "XFYUN_APP_ID"
XFYUN_ASR_DEFAULT_ACCESS_KEY_ID_ENV = "XFYUN_API_KEY"
XFYUN_ASR_DEFAULT_ACCESS_KEY_SECRET_ENV = "XFYUN_API_SECRET"
XFYUN_ASR_DEFAULT_LANGUAGE = "autodialect"
XFYUN_ASR_DEFAULT_RESULT_TYPE = "transfer"
XFYUN_ASR_DEFAULT_MAX_FILE_BYTES = 500_000_000
BAIDU_ASR_DEFAULT_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
BAIDU_ASR_DEFAULT_API_BASE_URL = "https://vop.baidu.com"
BAIDU_ASR_DEFAULT_API_KEY_ENV = "BAIDU_ASR_API_KEY"
BAIDU_ASR_DEFAULT_SECRET_KEY_ENV = "BAIDU_ASR_SECRET_KEY"
BAIDU_ASR_DEFAULT_CUID = "vidwise"
BAIDU_ASR_DEFAULT_DEV_PID = 1537
BAIDU_ASR_DEFAULT_RATE = 16000
BAIDU_ASR_DEFAULT_CHANNEL = 1
BAIDU_ASR_DEFAULT_CHUNK_SECONDS = 55.0
BAIDU_ASR_DEFAULT_MAX_CHUNK_BYTES = 10_000_000


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


def _clamped_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


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


class XFYunASRBackend:
    """iFLYTEK Spark recording-file transcription model."""

    def __init__(
        self,
        model_config: dict[str, Any],
        transcribe_config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
        sleep: Any = time.sleep,
    ) -> None:
        del transcribe_config

        configured_base_url = str(model_config.get("api_base_url") or "").strip()
        self.api_base_url = str(
            model_config.get("xfyun_api_base_url")
            or (
                configured_base_url
                if configured_base_url and configured_base_url != ALIYUN_ASR_DEFAULT_BASE_URL
                else ""
            )
            or XFYUN_ASR_DEFAULT_BASE_URL
        ).rstrip("/")
        self.app_id = _resolve_config_value(
            model_config,
            "xfyun_app_id",
            "xfyun_app_id_env",
            XFYUN_ASR_DEFAULT_APP_ID_ENV,
        )
        self.access_key_id = _resolve_config_value(
            model_config,
            "xfyun_access_key_id",
            "xfyun_access_key_id_env",
            XFYUN_ASR_DEFAULT_ACCESS_KEY_ID_ENV,
        )
        self.access_key_secret = _resolve_config_value(
            model_config,
            "xfyun_access_key_secret",
            "xfyun_access_key_secret_env",
            XFYUN_ASR_DEFAULT_ACCESS_KEY_SECRET_ENV,
        )
        missing = [
            name
            for name, value in (
                ("asr.model.xfyun_app_id", self.app_id),
                ("asr.model.xfyun_access_key_id", self.access_key_id),
                ("asr.model.xfyun_access_key_secret", self.access_key_secret),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} is required for xfyun provider")

        self.timeout = float(
            model_config.get("xfyun_api_timeout_seconds")
            or model_config.get("api_timeout_seconds")
            or 300
        )
        self.poll_interval_seconds = float(model_config.get("xfyun_poll_interval_seconds") or 3)
        self.max_poll_seconds = float(model_config.get("xfyun_max_poll_seconds") or 600)
        self.max_file_bytes = int(model_config.get("xfyun_max_file_bytes") or XFYUN_ASR_DEFAULT_MAX_FILE_BYTES)
        self.language = str(model_config.get("xfyun_language") or XFYUN_ASR_DEFAULT_LANGUAGE)
        self.result_type = str(model_config.get("xfyun_result_type") or XFYUN_ASR_DEFAULT_RESULT_TYPE)
        self.duration_check_disable = bool(model_config.get("xfyun_duration_check_disable", True))
        self.urlopen = urlopen
        self.sleep = sleep

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
        del beam_size, vad_filter, initial_prompt

        order_id, duration = self._upload(audio, language, sample_rate)
        payload = self._wait_for_result(order_id)
        segments = _parse_xfyun_segments(payload)
        text = "\n".join(segment.text for segment in segments if segment.text).strip()
        if segments:
            duration = max(duration, max(segment.end for segment in segments))
        return TranscriptionResult(
            text=text,
            language=language or self.language or "auto",
            language_probability=1.0,
            duration=duration,
            segments=segments,
        )

    def _upload(self, audio: str | np.ndarray, language: str, sample_rate: int) -> tuple[str, float]:
        if isinstance(audio, np.ndarray):
            file_name = "audio.wav"
            file_bytes = _ndarray_to_wav_bytes(audio, sample_rate)
            duration = float(audio.size) / float(sample_rate)
            params = self._upload_file_params(file_name, len(file_bytes), language)
            response = self._post("/upload", params, file_bytes, "application/octet-stream")
        elif audio.startswith(("http://", "https://")):
            duration = 0.0
            params = self._common_params()
            params.update(
                {
                    "uploadMode": "urlLink",
                    "fileUrl": audio,
                    "language": self._request_language(language),
                    "durationCheckDisable": _bool_query(self.duration_check_disable),
                }
            )
            response = self._post("/upload", params, None, "application/json;charset=UTF-8")
        else:
            file_name = os.path.basename(audio) or "audio"
            stat = os.stat(audio)
            if stat.st_size > self.max_file_bytes:
                raise ValueError(
                    f"audio file is {stat.st_size} bytes, exceeds xfyun limit "
                    f"configured by asr.model.xfyun_max_file_bytes={self.max_file_bytes}"
                )
            with open(audio, "rb") as f:
                file_bytes = f.read()
            duration = _wav_duration(audio)
            params = self._upload_file_params(file_name, stat.st_size, language)
            response = self._post("/upload", params, file_bytes, "application/octet-stream")

        content = response.get("content") or {}
        order_id = content.get("orderId") or response.get("orderId")
        if not order_id:
            raise RuntimeError(f"xfyun upload response did not include orderId: {response}")
        return str(order_id), duration

    def _wait_for_result(self, order_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.max_poll_seconds
        last_response: dict[str, Any] | None = None
        while True:
            response = self._post(
                "/getResult",
                self._result_params(order_id),
                None,
                "application/json;charset=UTF-8",
            )
            last_response = response
            content = response.get("content") or {}
            order_info = content.get("orderInfo") or {}
            status = str(order_info.get("status", ""))
            fail_type = str(order_info.get("failType", "0"))
            order_result = content.get("orderResult") or response.get("orderResult")

            if order_result:
                return _decode_xfyun_order_result(order_result)
            if status in {"-1", "5"} or (fail_type not in {"", "0", "None"} and status not in {"3", "4"}):
                raise RuntimeError(f"xfyun transcription failed: {response}")
            if time.monotonic() >= deadline:
                break
            self.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"xfyun transcription timed out waiting for orderId={order_id}: {last_response}")

    def _post(
        self,
        path: str,
        params: dict[str, str],
        data: bytes | None,
        content_type: str,
    ) -> dict[str, Any]:
        query = _canonical_query(params)
        signature = self._signature(query)
        request = urllib.request.Request(
            self.api_base_url + path + "?" + query,
            data=data,
            method="POST",
            headers={
                "Content-Type": content_type,
                "signature": signature,
            },
        )
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"xfyun asr returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"call xfyun asr failed: {exc.reason}") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("decode xfyun asr response failed") from exc

        code = str(payload.get("code") or "")
        if code and code != "000000":
            raise RuntimeError(f"xfyun asr returned code {code}: {payload}")
        return payload

    def _common_params(self) -> dict[str, str]:
        return {
            "appId": self.app_id,
            "accessKeyId": self.access_key_id,
            "dateTime": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "ts": str(int(time.time() * 1000)),
            "signa": secrets.token_hex(8),
        }

    def _upload_file_params(self, file_name: str, file_size: int, language: str) -> dict[str, str]:
        params = self._common_params()
        params.update(
            {
                "uploadMode": "fileStream",
                "fileName": file_name,
                "fileSize": str(file_size),
                "language": self._request_language(language),
                "durationCheckDisable": _bool_query(self.duration_check_disable),
            }
        )
        return params

    def _result_params(self, order_id: str) -> dict[str, str]:
        params = self._common_params()
        params.update({"orderId": order_id, "resultType": self.result_type})
        return params

    def _request_language(self, language: str) -> str:
        language = (language or "").strip().lower()
        if language in {"zh", "zh-cn", "cn", "auto", ""}:
            return self.language
        return language

    def _signature(self, query: str) -> str:
        digest = hmac.new(self.access_key_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha1).digest()
        return base64.b64encode(digest).decode("ascii")


class BaiduASRBackend:
    """Baidu Cloud short speech REST API with local-file WAV chunking."""

    def __init__(
        self,
        model_config: dict[str, Any],
        transcribe_config: dict[str, Any] | None = None,
        urlopen: Any = urllib.request.urlopen,
        audio_converter: Any = None,
    ) -> None:
        del transcribe_config

        configured_base_url = str(model_config.get("api_base_url") or "").strip()
        self.api_base_url = str(
            model_config.get("baidu_api_base_url")
            or (
                configured_base_url
                if configured_base_url and configured_base_url != ALIYUN_ASR_DEFAULT_BASE_URL
                else ""
            )
            or BAIDU_ASR_DEFAULT_API_BASE_URL
        ).rstrip("/")
        self.token_url = str(model_config.get("baidu_token_url") or BAIDU_ASR_DEFAULT_TOKEN_URL).strip()
        self.api_key = _resolve_baidu_api_key(model_config)
        self.secret_key = _resolve_config_value(
            model_config,
            "baidu_secret_key",
            "baidu_secret_key_env",
            BAIDU_ASR_DEFAULT_SECRET_KEY_ENV,
        )
        missing = [
            name
            for name, value in (
                ("asr.model.baidu_api_key", self.api_key),
                ("asr.model.baidu_secret_key", self.secret_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} is required for baidu provider")

        self.timeout = float(
            model_config.get("baidu_api_timeout_seconds")
            or model_config.get("api_timeout_seconds")
            or 60
        )
        self.dev_pid = int(model_config.get("baidu_dev_pid") or BAIDU_ASR_DEFAULT_DEV_PID)
        self.rate = int(model_config.get("baidu_rate") or BAIDU_ASR_DEFAULT_RATE)
        self.channel = int(model_config.get("baidu_channel") or BAIDU_ASR_DEFAULT_CHANNEL)
        self.cuid = str(model_config.get("baidu_cuid") or BAIDU_ASR_DEFAULT_CUID)
        self.chunk_seconds = _clamped_float(
            model_config.get("baidu_chunk_seconds"),
            BAIDU_ASR_DEFAULT_CHUNK_SECONDS,
            1.0,
            60.0,
        )
        self.max_chunk_bytes = int(
            model_config.get("baidu_max_chunk_bytes") or BAIDU_ASR_DEFAULT_MAX_CHUNK_BYTES
        )
        self.urlopen = urlopen
        self.audio_converter = audio_converter or _ffmpeg_wav_chunks
        self._access_token = ""
        self._token_expiry_monotonic = 0.0

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
        del beam_size, vad_filter, initial_prompt

        segments: list[TranscriptionSegment] = []
        texts: list[str] = []
        offset = 0.0
        for chunk_bytes, duration, rate in self._audio_chunks(audio, sample_rate):
            if len(chunk_bytes) > self.max_chunk_bytes:
                raise ValueError(
                    f"baidu asr chunk is {len(chunk_bytes)} bytes, exceeds limit "
                    f"configured by asr.model.baidu_max_chunk_bytes={self.max_chunk_bytes}"
                )
            text = self._recognize_chunk(chunk_bytes, rate).strip()
            end = offset + duration
            if text:
                texts.append(text)
                segments.append(TranscriptionSegment(start=offset, end=end, text=text))
            offset = end

        return TranscriptionResult(
            text="\n".join(texts).strip(),
            language=language or "zh",
            language_probability=1.0,
            duration=offset,
            segments=segments,
        )

    def _audio_chunks(self, audio: str | np.ndarray, sample_rate: int):
        if isinstance(audio, np.ndarray):
            yield from _ndarray_wav_chunks(audio, sample_rate, self.chunk_seconds)
            return
        if audio.startswith(("http://", "https://")):
            raise ValueError(
                "baidu short speech REST API requires local audio bytes; "
                "configure a file-URL transcription flow for remote URLs"
            )
        yield from self.audio_converter(audio, self.rate, self.chunk_seconds)

    def _recognize_chunk(self, chunk_bytes: bytes, rate: int) -> str:
        payload = {
            "format": "wav",
            "rate": int(rate),
            "channel": self.channel,
            "cuid": self.cuid,
            "token": self._access_token_value(),
            "dev_pid": self.dev_pid,
            "speech": base64.b64encode(chunk_bytes).decode("ascii"),
            "len": len(chunk_bytes),
        }
        response = self._post_json(self.api_base_url + "/server_api", payload)
        return _parse_baidu_asr_response(response)

    def _access_token_value(self) -> str:
        now = time.monotonic()
        if self._access_token and now < self._token_expiry_monotonic:
            return self._access_token

        params = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            }
        )
        request = urllib.request.Request(
            self.token_url + "?" + params,
            data=b"",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        payload = self._read_json(request, "baidu token")
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError(f"baidu token response did not include access_token: {payload}")

        try:
            expires_in = int(payload.get("expires_in") or 0)
        except (TypeError, ValueError):
            expires_in = 0
        self._access_token = access_token
        self._token_expiry_monotonic = now + max(60, expires_in - 60) if expires_in else now + 3600
        return access_token

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._read_json(request, "baidu asr")

    def _read_json(self, request: urllib.request.Request, label: str) -> dict[str, Any]:
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{label} returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"call {label} failed: {exc.reason}") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"decode {label} response failed") from exc


def _resolve_api_key(config: dict[str, Any], default_env: str) -> str:
    explicit = str(config.get("api_key") or "").strip()
    if explicit:
        return explicit
    env_name = str(config.get("api_key_env") or default_env).strip()
    if not env_name:
        return ""
    return os.getenv(env_name, "").strip()


def _resolve_baidu_api_key(config: dict[str, Any]) -> str:
    value = _resolve_config_value(
        config,
        "baidu_api_key",
        "baidu_api_key_env",
        BAIDU_ASR_DEFAULT_API_KEY_ENV,
    )
    if value:
        return value
    explicit = str(config.get("api_key") or "").strip()
    if explicit:
        return explicit
    env_name = str(config.get("api_key_env") or "").strip()
    if env_name and env_name != ALIYUN_ASR_DEFAULT_API_KEY_ENV:
        return os.getenv(env_name, "").strip()
    return ""


def _resolve_config_value(config: dict[str, Any], value_key: str, env_key: str, default_env: str) -> str:
    explicit = str(config.get(value_key) or "").strip()
    if explicit:
        return explicit
    env_name = str(config.get(env_key) or default_env).strip()
    if not env_name:
        return ""
    return os.getenv(env_name, "").strip()


def _aliyun_model_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    if not name or name.startswith((".", "/")):
        return ALIYUN_ASR_DEFAULT_MODEL
    return name


def _ndarray_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio_i16 = (np.clip(audio.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())
    return buffer.getvalue()


def _ndarray_to_wav_data_url(audio: np.ndarray, sample_rate: int) -> str:
    encoded = base64.b64encode(_ndarray_to_wav_bytes(audio, sample_rate)).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _ndarray_wav_chunks(audio: np.ndarray, sample_rate: int, chunk_seconds: float):
    audio_array = audio.astype(np.float32)
    samples_per_chunk = max(1, int(float(sample_rate) * chunk_seconds))
    for offset in range(0, audio_array.size, samples_per_chunk):
        chunk = audio_array[offset:offset + samples_per_chunk]
        if chunk.size == 0:
            continue
        yield (
            _ndarray_to_wav_bytes(chunk, sample_rate),
            float(chunk.size) / float(sample_rate),
            sample_rate,
        )


def _ffmpeg_wav_chunks(audio_path: str, rate: int, chunk_seconds: float):
    with tempfile.TemporaryDirectory(prefix="vidwise-baidu-asr-") as tmpdir:
        output_pattern = os.path.join(tmpdir, "chunk_%05d.wav")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            audio_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "segment",
            "-segment_time",
            f"{chunk_seconds:g}",
            "-reset_timestamps",
            "1",
            output_pattern,
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for baidu ASR local-file transcoding") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to prepare baidu ASR audio chunks: {detail}") from exc

        chunk_paths = sorted(
            os.path.join(tmpdir, name)
            for name in os.listdir(tmpdir)
            if name.endswith(".wav")
        )
        if not chunk_paths:
            raise RuntimeError("ffmpeg did not produce any baidu ASR audio chunks")
        for chunk_path in chunk_paths:
            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()
            yield chunk_bytes, _wav_duration(chunk_path), rate


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        return float(frames) / float(rate) if rate else 0.0
    except (wave.Error, OSError, EOFError):
        return 0.0


def _parse_baidu_asr_response(payload: dict[str, Any]) -> str:
    err_no = payload.get("err_no")
    if err_no is not None and str(err_no) != "0":
        raise RuntimeError(f"baidu asr returned err_no {err_no}: {payload}")
    if "result" not in payload:
        raise RuntimeError(f"baidu asr response did not include result: {payload}")
    result = payload.get("result")
    if isinstance(result, list):
        return "\n".join(str(item).strip() for item in result if str(item).strip()).strip()
    if isinstance(result, str):
        return result.strip()
    return ""


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


def _canonical_query(params: dict[str, str]) -> str:
    return urllib.parse.urlencode(
        [(key, str(params[key])) for key in sorted(params)],
        quote_via=urllib.parse.quote,
    )


def _bool_query(value: bool) -> str:
    return "true" if value else "false"


def _decode_xfyun_order_result(order_result: Any) -> dict[str, Any]:
    if isinstance(order_result, dict):
        return order_result
    if isinstance(order_result, str):
        try:
            return json.loads(order_result)
        except json.JSONDecodeError as exc:
            raise RuntimeError("decode xfyun orderResult failed") from exc
    raise RuntimeError(f"unsupported xfyun orderResult type: {type(order_result)}")


def _parse_xfyun_segments(order_result: dict[str, Any]) -> list[TranscriptionSegment]:
    segments: list[TranscriptionSegment] = []
    for item in order_result.get("lattice") or []:
        best = item.get("json_1best") or item.get("json_1Best")
        if isinstance(best, str):
            try:
                best = json.loads(best)
            except json.JSONDecodeError:
                best = {}
        if not isinstance(best, dict):
            continue

        st = best.get("st") or {}
        text = _xfyun_st_text(st).strip()
        if not text:
            continue
        start = _xfyun_time_seconds(st.get("bg") or item.get("begin") or item.get("bg"))
        end = _xfyun_time_seconds(st.get("ed") or item.get("end") or item.get("ed"))
        if end < start:
            end = start
        segments.append(TranscriptionSegment(start=start, end=end, text=text))
    return segments


def _xfyun_st_text(st: dict[str, Any]) -> str:
    parts: list[str] = []
    for rt in st.get("rt") or []:
        for ws in rt.get("ws") or []:
            candidates = ws.get("cw") or []
            if candidates:
                parts.append(str(candidates[0].get("w") or ""))
    return "".join(parts)


def _xfyun_time_seconds(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value / 1000.0 if value > 60 else value


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
    if provider in {"xfyun", "iflytek"}:
        return XFYunASRBackend(model_config, transcribe_config)
    if provider in {"baidu", "baiducloud", "baidu-cloud"}:
        return BaiduASRBackend(model_config, transcribe_config)
    raise ValueError(f"Unsupported ASR provider: {provider}")
