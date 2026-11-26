GO ?= go
VENV ?= .venv/bin/

ifeq ($(OS),Windows_NT)
PYTHON ?= python
VENVPYTHON ?= python
else
PYTHON ?= python3
VENVPYTHON ?= .venv/bin/python3
endif

.PHONY: help deps check-deps deps-python deps-embedding install-ffmpeg install-yt-dlp run run-asr run-embedding run-all-bg stop-all test

help:
	@echo "Usage:"
	@echo "  make deps               Check/install ffmpeg and yt-dlp"
	@echo "  make deps-python        Install Python ASR dependencies"
	@echo "  make deps-embedding     Install Python embedding service dependencies"
	@echo "  make run-asr            Run the ASR service (port 8001)"
	@echo "  make run-embedding      Run the embedding service (port 8003)"
	@echo "  make run                Run the Go gateway (port from config)"
	@echo "  make run-all-bg         Start all services in background"
	@echo "  make stop-all           Stop all services"
	@echo "  make test               Run Go tests"

deps: check-deps

check-deps: install-ffmpeg install-yt-dlp

deps-python:
	$(VENVPYTHON) -m pip install -r asr_service/requirements.txt

deps-embedding:
	$(VENVPYTHON) -m pip install -r embedding_service/requirements.txt

ifeq ($(OS),Windows_NT)
install-ffmpeg:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Write-Host 'ffmpeg already installed' } elseif (Get-Command winget -ErrorAction SilentlyContinue) { winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements } elseif (Get-Command choco -ErrorAction SilentlyContinue) { choco install ffmpeg -y } else { throw 'ffmpeg is missing. Install winget or Chocolatey, then run make deps again.' }"

install-yt-dlp:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-Command yt-dlp -ErrorAction SilentlyContinue) { Write-Host 'yt-dlp already installed' } elseif (Get-Command winget -ErrorAction SilentlyContinue) { winget install --id yt-dlp.yt-dlp -e --accept-package-agreements --accept-source-agreements } elseif (Get-Command choco -ErrorAction SilentlyContinue) { choco install yt-dlp -y } elseif (Get-Command python -ErrorAction SilentlyContinue) { python -m pip install --user --upgrade yt-dlp } else { throw 'yt-dlp is missing. Install winget, Chocolatey, or Python, then run make deps again.' }"
else
install-ffmpeg:
	@if command -v ffmpeg >/dev/null 2>&1; then \
		echo "ffmpeg already installed"; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install ffmpeg; \
	else \
		echo "ffmpeg is missing. Install Homebrew, then run make deps again."; \
		exit 1; \
	fi

install-yt-dlp:
	@if command -v yt-dlp >/dev/null 2>&1; then \
		echo "yt-dlp already installed"; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install yt-dlp; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 -m pip install --user --upgrade yt-dlp; \
	else \
		echo "yt-dlp is missing. Install Homebrew or Python 3, then run make deps again."; \
		exit 1; \
	fi

endif

run: deps
	$(GO) run . 

run-asr:
	$(VENVPYTHON) -m uvicorn asr_service.app:app --host 0.0.0.0 --port 8001

run-embedding:
	$(VENVPYTHON) -m uvicorn embedding_service.app:app --host 0.0.0.0 --port 8003

run-all-bg:
	@echo "Starting all services in background..."
	@echo "  Embedding (8003)..."
	@$(VENVPYTHON) -m uvicorn embedding_service.app:app --host 0.0.0.0 --port 8003 --log-level warning &
	@sleep 3
	@echo "  ASR (8001)..."
	@$(VENVPYTHON) -m uvicorn asr_service.app:app --host 0.0.0.0 --port 8001 --log-level warning &
	@sleep 3
	@echo "  Gateway (8080)..."
	@$(GO) run . &
	@sleep 2
	@echo "All services started!"
	@echo "  Embedding: http://localhost:8003/health"
	@echo "  ASR:       http://localhost:8001/health"
	@echo "  Gateway:   http://localhost:8080"

stop-all:
	@echo "Stopping all services..."
	@lsof -ti:8001 -ti:8003 -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null || true
	@echo "All stopped."

test:
	$(GO) test ./...