# ASR Service (Streaming)

This service exposes ASR HTTP and streaming WebSocket endpoints backed by a pluggable ASR backend.

## Backends

Set the backend in `config.yaml`:

```yaml
asr:
  model:
    provider: "faster-whisper"
    name: "small"
```

For SenseVoice:

```yaml
asr:
  model:
    provider: "sensevoice"
    name: "FunAudioLLM/SenseVoiceSmall"
```

The service keeps the same JSON response shape for all backends:

- `text`
- `language`
- `language_probability`
- `duration`
- `segments`

SenseVoice does not currently return word or segment timestamps through this adapter, so the response contains one segment covering the whole input when duration can be determined.

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
uvicorn app:app --host 0.0.0.0 --port 8001
```

2. Stream a 16 kHz mono WAV file:

```bash
python stream_client.py --ws-url ws://127.0.0.1:8001/stream --wav ./samples/16k_mono.wav
```

## Notes

- On first run, the Silero VAD weights are downloaded via `torch.hub`.
- Adjust stream settings in `config.yaml` under `asr.stream`.
