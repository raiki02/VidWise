# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

Minimal runnable MVP of a video intelligence platform. A Go/Gin gateway (`main.go`, `internal/server/`) hosts a **CloudWeGo eino ReAct Agent** that orchestrates the full RAG loop:

- **Agent & tools** — `internal/agent/` builds `react.NewAgent` with the tools from `internal/tool/`: `download_audio` (yt-dlp via `cmd/download`), `transcribe_audio` (calls the ASR service through `internal/asr`), `index_transcript` (chunk via `internal/rag` → embed via `internal/embedding` → upsert into Qdrant via `internal/vectorstore`), `search_knowledge` (embed query → vector search).
- **ASR** — Python FastAPI service (`asr_service/`, port 8001) exposing `POST /transcribe` and `GET /health`. Providers: whisper / faster-whisper (local), aliyun / xfyun / baidu (cloud). The Go side calls it via `internal/asr/client.go`.
- **Embedding** — Python FastAPI service (`embedding_service/`, port 8003) exposing `POST /embed` and `GET /health`. Providers: local (sentence-transformers), aliyun, siliconflow. No rerank in the MVP.
- **Vector store** — Qdrant gRPC (port 6334) via `internal/vectorstore`; collection `transcript_chunks`, Cosine distance, source_id payload for per-video replace.
- **Chat sessions** — `internal/chat/` Service: create/find session, load history, call the agent, persist both messages. Persistence via MySQL/GORM (`internal/storage/mysql`); when `mysql.dsn` is empty it falls back to an in-memory store (`internal/chat/memory.go`) that is lost on restart.

Removed subsystems (do not resurrect without being asked): web search, MCP, video summary, rerank, LLM paragraph formatting, async task queue, capability manifest endpoints.

## Request flow

```
POST /agent/chat {session_id?, message}
  → server.ChatHandler.AgentChat validates (400 on bad JSON/empty message)
  → chat.Service.Chat: resolve session (404 if unknown) → load history → agent.Generate
  → ReAct loop: LLM decides tool calls (download_audio → transcribe_audio →
     index_transcript / search_knowledge)
  → persist user + assistant messages on success only
  → {session_id, reply}
```

`GET /` serves the embedded single-page UI (`internal/server/web/index.html`); `GET /health` is a plain liveness check.

## Commands

```bash
make run                # Start Go gateway (port from config, default :8080)
make run-asr            # Start Python ASR service (uvicorn, port 8001)
make run-embedding      # Start Python embedding service (uvicorn, port 8003)
make run-all-bg         # Start all three services in the background
make stop-all           # Kill processes on 8001/8003/8080
make test               # Run all Go tests (go test ./...)
make deps               # Check/install ffmpeg and yt-dlp
make deps-python        # pip install asr_service/requirements.txt
make deps-embedding     # pip install embedding_service/requirements.txt
```

The Go gateway requires LLM + ASR + Embedding + Qdrant to be reachable at startup (fatal on failure); MySQL is optional.

## Configuration

Copy `config.example.yaml` to `config.yaml` (gitignored). Key sections:

- `server.addr` — listen address (`write_timeout: 0s` is intentional: long tool turns)
- `download.cookies_path`, `download.work_dir` — yt-dlp cookies; sandbox dir for media/transcripts
- `asr.base_url` / `asr.model.*` — ASR adapter address + provider (whisper/faster-whisper/aliyun/xfyun/baidu)
- `embedding.base_url` / `embedding.*` — embedding adapter address + provider (local/aliyun/siliconflow); `dimensions` must match `qdrant.vector_dim`
- `llm.provider/model/api_key` — `openai`, `ollama`, or `deepseek` (eino-ext)
- `qdrant.*` — host/port/collection/vector_dim
- `rag.chunk_runes` / `overlap_runes` / `top_k`
- `agent.max_step` / `history_limit`
- `mysql.dsn` — optional; empty = in-memory chat history

## Key dependencies

- **Go**: Gin (HTTP), CloudWeGo Eino (react agent / tool), eino-ext chat models, Qdrant gRPC client, GORM (only when MySQL configured)
- **Python ASR**: fastapi, uvicorn, transformers, torch, faster-whisper, librosa (whisper backend)
- **Python embedding**: fastapi, uvicorn, sentence-transformers, torch, pyyaml
- **System**: ffmpeg, yt-dlp