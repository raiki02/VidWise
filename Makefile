GO ?= go

ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= python3
endif

.PHONY: help deps check-deps deps-python install-ffmpeg install-yt-dlp install-yd-dlp check-whisper-cli run run-asr test

help:
	@echo "Usage:"
	@echo "  make deps        Check/install ffmpeg and yt-dlp, and check whisper-cli fallback"
	@echo "  make deps-python Install Python ASR dependencies"
	@echo "  make run-asr     Run the faster-whisper ASR service"
	@echo "  make run         Install deps when needed, then run the Go backend service"
	@echo "  make test        Run Go tests"

deps: check-deps

check-deps: install-ffmpeg install-yt-dlp check-whisper-cli

deps-python:
	$(PYTHON) -m pip install -r asr_service/requirements.txt

ifeq ($(OS),Windows_NT)
install-ffmpeg:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Write-Host 'ffmpeg already installed' } elseif (Get-Command winget -ErrorAction SilentlyContinue) { winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements } elseif (Get-Command choco -ErrorAction SilentlyContinue) { choco install ffmpeg -y } else { throw 'ffmpeg is missing. Install winget or Chocolatey, then run make deps again.' }"

install-yt-dlp:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-Command yt-dlp -ErrorAction SilentlyContinue) { Write-Host 'yt-dlp already installed' } elseif (Get-Command winget -ErrorAction SilentlyContinue) { winget install --id yt-dlp.yt-dlp -e --accept-package-agreements --accept-source-agreements } elseif (Get-Command choco -ErrorAction SilentlyContinue) { choco install yt-dlp -y } elseif (Get-Command python -ErrorAction SilentlyContinue) { python -m pip install --user --upgrade yt-dlp } else { throw 'yt-dlp is missing. Install winget, Chocolatey, or Python, then run make deps again.' }"

check-whisper-cli:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-Command whisper-cli -ErrorAction SilentlyContinue) { Write-Host 'whisper-cli already installed' } else { throw 'whisper-cli is missing. It is required only as fallback when the Python ASR service is unavailable. Install whisper.cpp and make sure whisper-cli is on PATH.' }"
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

check-whisper-cli:
	@if command -v whisper-cli >/dev/null 2>&1; then \
		echo "whisper-cli already installed"; \
	else \
		echo "whisper-cli is missing. It is required only as fallback when the Python ASR service is unavailable."; \
		echo "Install whisper.cpp and make sure whisper-cli is on PATH."; \
		exit 1; \
	fi
endif

install-yd-dlp: install-yt-dlp

run: deps
	$(GO) run .

run-asr:
	$(PYTHON) -m uvicorn asr_service.app:app --host 0.0.0.0 --port 8001

test:
	$(GO) test ./...
