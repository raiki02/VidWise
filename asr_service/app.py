import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure the asr_service directory is on the path for local imports.
_svc_dir = os.path.dirname(os.path.abspath(__file__))
if _svc_dir not in sys.path:
    sys.path.insert(0, _svc_dir)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import yaml

from backends import ASRBackend, TranscriptionResult, create_asr_backend


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.yaml"))
DEFAULT_ASR_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "whisper",
        "name": "./models/whisper-small",
        "device": "auto",
        "torch_dtype": "auto",
        "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_timeout_seconds": 300,
        "max_file_bytes": 7500000,
        "xfyun_api_base_url": "https://office-api-ist-dx.iflyaisol.com/v2",
        "xfyun_app_id": "",
        "xfyun_app_id_env": "XFYUN_APP_ID",
        "xfyun_access_key_id": "",
        "xfyun_access_key_id_env": "XFYUN_API_KEY",
        "xfyun_access_key_secret": "",
        "xfyun_access_key_secret_env": "XFYUN_API_SECRET",
        "xfyun_language": "autodialect",
        "xfyun_result_type": "transfer",
        "xfyun_api_timeout_seconds": 300,
        "xfyun_poll_interval_seconds": 3,
        "xfyun_max_poll_seconds": 600,
        "xfyun_max_file_bytes": 500000000,
        "baidu_token_url": "https://aip.baidubce.com/oauth/2.0/token",
        "baidu_api_base_url": "https://vop.baidu.com",
        "baidu_api_key": "",
        "baidu_api_key_env": "BAIDU_ASR_API_KEY",
        "baidu_secret_key": "",
        "baidu_secret_key_env": "BAIDU_ASR_SECRET_KEY",
        "baidu_cuid": "vidwise",
        "baidu_dev_pid": 1537,
        "baidu_rate": 16000,
        "baidu_channel": 1,
        "baidu_api_timeout_seconds": 60,
        "baidu_chunk_seconds": 55,
        "baidu_max_chunk_bytes": 10000000,
    },
    "transcribe": {
        "beam_size": 5,
        "vad_filter": True,  # accepted for compatibility; VAD is not applied
        "initial_prompt": "",
        "max_new_tokens": 448,
        "max_initial_prompt_tokens": 224,
    },
}

app = FastAPI(title="vidwise-asr")
model: ASRBackend | None = None
asr_config: dict[str, Any] = DEFAULT_ASR_CONFIG


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("vidwise.asr")


@app.on_event("startup")
def load_model() -> None:
    global asr_config, model

    try:
        asr_config = load_asr_config()
        model_config = asr_config["model"]
        logger.info(
            "loading ASR model provider=%s model=%s device=%s",
            model_config.get("provider"),
            model_config.get("name"),
            model_config.get("device", ""),
        )
        model = create_asr_backend(
            model_config,
            transcribe_config=asr_config["transcribe"],
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
    if not initial_prompt:
        initial_prompt = transcribe_config["initial_prompt"]
    # vad_filter is accepted for Go client compatibility but ignored: the MVP
    # backend implementations do not apply VAD.

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
            initial_prompt=initial_prompt or None,
        )
    except Exception as exc:
        logger.exception("transcribe failed filename=%s language=%s beam_size=%s", file.filename, language, beam_size)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return _result_payload(result)


def load_asr_config() -> dict[str, Any]:
    config = merge_dict(DEFAULT_ASR_CONFIG, {})
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = merge_dict(config, data.get("asr") or {})

    model_config = config["model"]
    transcribe_config = config["transcribe"]

    model_config["provider"] = os.getenv("ASR_PROVIDER", model_config.get("provider", "whisper"))
    model_config["name"] = os.getenv("ASR_MODEL", model_config["name"])
    model_config["device"] = os.getenv("ASR_DEVICE", model_config["device"])
    model_config["torch_dtype"] = os.getenv("ASR_TORCH_DTYPE", model_config.get("torch_dtype", "auto"))
    model_config["api_base_url"] = os.getenv("ASR_API_BASE_URL", model_config.get("api_base_url", ""))
    model_config["api_key"] = os.getenv("ASR_API_KEY", model_config.get("api_key", ""))
    model_config["api_key_env"] = os.getenv("ASR_API_KEY_ENV", model_config.get("api_key_env", "DASHSCOPE_API_KEY"))
    model_config["api_timeout_seconds"] = float(
        os.getenv("ASR_API_TIMEOUT_SECONDS", model_config.get("api_timeout_seconds", 300))
    )
    model_config["max_file_bytes"] = int(
        os.getenv("ASR_MAX_FILE_BYTES", model_config.get("max_file_bytes", 7500000))
    )
    model_config["xfyun_api_base_url"] = os.getenv(
        "ASR_XFYUN_API_BASE_URL",
        model_config.get("xfyun_api_base_url", "https://office-api-ist-dx.iflyaisol.com/v2"),
    )
    model_config["xfyun_app_id"] = os.getenv("ASR_XFYUN_APP_ID", model_config.get("xfyun_app_id", ""))
    model_config["xfyun_app_id_env"] = os.getenv(
        "ASR_XFYUN_APP_ID_ENV",
        model_config.get("xfyun_app_id_env", "XFYUN_APP_ID"),
    )
    model_config["xfyun_access_key_id"] = os.getenv(
        "ASR_XFYUN_ACCESS_KEY_ID",
        model_config.get("xfyun_access_key_id", ""),
    )
    model_config["xfyun_access_key_id_env"] = os.getenv(
        "ASR_XFYUN_ACCESS_KEY_ID_ENV",
        model_config.get("xfyun_access_key_id_env", "XFYUN_API_KEY"),
    )
    model_config["xfyun_access_key_secret"] = os.getenv(
        "ASR_XFYUN_ACCESS_KEY_SECRET",
        model_config.get("xfyun_access_key_secret", ""),
    )
    model_config["xfyun_access_key_secret_env"] = os.getenv(
        "ASR_XFYUN_ACCESS_KEY_SECRET_ENV",
        model_config.get("xfyun_access_key_secret_env", "XFYUN_API_SECRET"),
    )
    model_config["xfyun_language"] = os.getenv(
        "ASR_XFYUN_LANGUAGE",
        model_config.get("xfyun_language", "autodialect"),
    )
    model_config["xfyun_result_type"] = os.getenv(
        "ASR_XFYUN_RESULT_TYPE",
        model_config.get("xfyun_result_type", "transfer"),
    )
    model_config["xfyun_api_timeout_seconds"] = float(
        os.getenv("ASR_XFYUN_API_TIMEOUT_SECONDS", model_config.get("xfyun_api_timeout_seconds", 300))
    )
    model_config["xfyun_poll_interval_seconds"] = float(
        os.getenv("ASR_XFYUN_POLL_INTERVAL_SECONDS", model_config.get("xfyun_poll_interval_seconds", 3))
    )
    model_config["xfyun_max_poll_seconds"] = float(
        os.getenv("ASR_XFYUN_MAX_POLL_SECONDS", model_config.get("xfyun_max_poll_seconds", 600))
    )
    model_config["xfyun_max_file_bytes"] = int(
        os.getenv("ASR_XFYUN_MAX_FILE_BYTES", model_config.get("xfyun_max_file_bytes", 500000000))
    )
    model_config["baidu_token_url"] = os.getenv(
        "ASR_BAIDU_TOKEN_URL",
        model_config.get("baidu_token_url", "https://aip.baidubce.com/oauth/2.0/token"),
    )
    model_config["baidu_api_base_url"] = os.getenv(
        "ASR_BAIDU_API_BASE_URL",
        model_config.get("baidu_api_base_url", "https://vop.baidu.com"),
    )
    model_config["baidu_api_key"] = os.getenv("ASR_BAIDU_API_KEY", model_config.get("baidu_api_key", ""))
    model_config["baidu_api_key_env"] = os.getenv(
        "ASR_BAIDU_API_KEY_ENV",
        model_config.get("baidu_api_key_env", "BAIDU_ASR_API_KEY"),
    )
    model_config["baidu_secret_key"] = os.getenv(
        "ASR_BAIDU_SECRET_KEY",
        model_config.get("baidu_secret_key", ""),
    )
    model_config["baidu_secret_key_env"] = os.getenv(
        "ASR_BAIDU_SECRET_KEY_ENV",
        model_config.get("baidu_secret_key_env", "BAIDU_ASR_SECRET_KEY"),
    )
    model_config["baidu_cuid"] = os.getenv("ASR_BAIDU_CUID", model_config.get("baidu_cuid", "vidwise"))
    model_config["baidu_dev_pid"] = int(
        os.getenv("ASR_BAIDU_DEV_PID", model_config.get("baidu_dev_pid", 1537))
    )
    model_config["baidu_rate"] = int(
        os.getenv("ASR_BAIDU_RATE", model_config.get("baidu_rate", 16000))
    )
    model_config["baidu_channel"] = int(
        os.getenv("ASR_BAIDU_CHANNEL", model_config.get("baidu_channel", 1))
    )
    model_config["baidu_api_timeout_seconds"] = float(
        os.getenv("ASR_BAIDU_API_TIMEOUT_SECONDS", model_config.get("baidu_api_timeout_seconds", 60))
    )
    model_config["baidu_chunk_seconds"] = float(
        os.getenv("ASR_BAIDU_CHUNK_SECONDS", model_config.get("baidu_chunk_seconds", 55))
    )
    model_config["baidu_max_chunk_bytes"] = int(
        os.getenv("ASR_BAIDU_MAX_CHUNK_BYTES", model_config.get("baidu_max_chunk_bytes", 10000000))
    )

    transcribe_config["beam_size"] = int(transcribe_config["beam_size"])
    transcribe_config["vad_filter"] = bool(transcribe_config["vad_filter"])
    transcribe_config["initial_prompt"] = transcribe_config["initial_prompt"] or ""
    transcribe_config["max_new_tokens"] = int(transcribe_config.get("max_new_tokens", 448))
    transcribe_config["max_initial_prompt_tokens"] = int(transcribe_config.get("max_initial_prompt_tokens", 224))

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