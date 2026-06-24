# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

This is a video/audio/text extraction service. The core is a Go/Gin HTTP server (`main.go`, `internal/server/`) that orchestrates three subsystems:

- **Download & transcode** — `cmd/download/`, `cmd/audio/`, `cmd/video/` wrap `yt-dlp` and `ffmpeg` as CLI subprocesses.
- **ASR (speech-to-text)** — Python FastAPI service (`asr_service/`) exposing `/transcribe` and `/stream` (WebSocket). Supports two backends: `WhisperBackend` (transformers) and `FasterWhisperBackend` (CTranslate2). The Go side calls it via an HTTP client (`internal/asr/client.go`) wrapped as an Eino tool (`internal/asr/tool.go`).
- **Video understanding** — Python FastAPI service (`video_summary_service/`) exposing `/caption` and `/find` endpoints backed by NemoStation/Marlin-2B (HF transformers or MLX). The Go side mirrors this in `internal/video_summary/`.
- **LLM paragraph formatting** — After ASR, raw transcript text is optionally formatted by an LLM. `internal/paragraph/chatmodel.go` builds a CloudWeGo Eino chat model for OpenAI, Ollama, or DeepSeek. `internal/paragraph/paragraph.go` splits long text into chunks, processes them in parallel (max 3 concurrent), and supports a `two_step` mode (step 1: per-chunk typo fix + trad→simp; step 2: semantic paragraph organization of the joined step-1 output).

## Request flow

```
HTTP GET/POST /extract
  → server.extract() binds url/name/type
  → extractor.Service.Extract() dispatches by type:
      video:  download.Video → video.Compatible (ffmpeg re-encode)
      audio:  download.Audio (yt-dlp -x)
      text:   download.Audio → agent.TranscriptAgent (ASR call → paragraph.FormatText)
      summary: download.Video → agent.VideoSummaryAgent (Marlin caption)
  → response is a file attachment
```

`/health`, `/format` (standalone text formatting endpoint), and `web/` (embedded HTML UI) are also served by Gin.

Eino `tool.InvokableTool` wrappers (`internal/asr/tool.go`, `internal/video_summary/tool.go`) exist so the agents could be used within an Eino graph, though currently they're invoked directly.

## Commands

```bash
make run                # Start Go backend (port from config, default :8080)
make run-asr            # Start Python ASR service (uvicorn, port 8001)
make run-video-summary  # Start Python video summary service (uvicorn, port 8002)
make test               # Run all Go tests (go test ./...)
make deps               # Install ffmpeg and yt-dlp
make deps-python        # pip install asr_service/requirements.txt
make deps-video-summary # pip install video_summary_service/requirements.txt
```

All three services must run concurrently for full functionality (Go backend + both Python services). The Python services read `config.yaml` for model/transcribe settings and also accept overrides from environment variables (`ASR_MODEL`, `ASR_DEVICE`, `VIDEO_SUMMARY_MODEL`, etc.).

## Configuration

Copy `config.example.yaml` to `config.yaml` (gitignored). Key sections:

- `server.addr` — listen address
- `download.cookies_path` — optional yt-dlp cookies file
- `asr.model.provider` — `whisper` or `faster-whisper`; `name` is a local path or HF model ID
- `video_summary.model.provider` — `huggingface` or `mlx` (Apple Silicon)
- `llm.enabled` — when false, returns raw ASR text without LLM formatting
- `llm.provider` — `openai`, `ollama`, or `deepseek`
- `llm.two_step` — enables the two-pass formatting pipeline
- `llm.fallback_to_raw_on_error` — when true, returns raw ASR text if the LLM is unavailable

Default prompts for step1 and step2 are defined as Go constants in `internal/appconfig/config.go`.

## Key dependencies

- **Go**: Gin (HTTP), CloudWeGo Eino (tool/model abstraction), gopkg.in/yaml.v3
- **Python ASR**: transformers, torch, faster-whisper, silero-vad, librosa, fastapi, uvicorn
- **Python video summary**: transformers, torch, opencv-python, fastapi, uvicorn; optionally mlx-vlm for Apple Silicon
- **System**: ffmpeg, yt-dlp (installed via `make deps`)