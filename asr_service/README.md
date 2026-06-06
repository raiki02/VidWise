# ASR Service (Streaming)

This service exposes ASR HTTP and streaming WebSocket endpoints backed by OpenAI Whisper via transformers.

## Backend

The service uses `transformers` with the Whisper model:

```yaml
asr:
  model:
    provider: "whisper"
    name: "./models/whisper-small"
    device: "auto"
    torch_dtype: "auto"
```

The `name` supports any HuggingFace Whisper model ID or local path (e.g., `"openai/whisper-small"`, `"openai/whisper-large-v3"`, or `"./models/whisper-small"`).

## Response Shape

All endpoints return:

- `text`
- `language`
- `language_probability`
- `duration`
- `segments`

## Streaming Endpoint

- URL: `ws://<host>:<port>/stream`
- Audio format: 16 kHz, mono, 16-bit PCM (little-endian)
- Messages:
  - Text JSON:
    - `{"event":"config","language":"zh","beam_size":1,"initial_prompt":"...","vad":{...}}`
    - `{"event":"end"}` to flush remaining audio and close
  - Binary: raw PCM chunks

Server responses:
- `{"type":"ready", ...}` after connection
- `{"type":"final", "id": 0, "text": "...", "segments": [...]}` when a speech segment ends
- `{"type":"error", "detail": "..."}` on errors

## Quick Try

1. Start the service (example):

```bash
uvicorn asr_service.app:app --host 0.0.0.0 --port 8001
```

2. Stream a 16 kHz mono WAV file:

```bash
python asr_service/stream_client.py --ws-url ws://127.0.0.1:8001/stream --wav ./samples/16k_mono.wav
```

## Notes

- On first run, the Whisper model weights are downloaded from Hugging Face (or loaded from the local path).
- The Silero VAD weights are downloaded via `torch.hub` on first use.
- Adjust stream settings in `config.yaml` under `asr.stream`.
