GO ?= go

ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= python3
endif

.PHONY: help deps check-deps deps-python deps-video-summary install-ffmpeg install-yt-dlp run run-asr run-video-summary test

help:
	@echo "Usage:"
	@echo "  make deps        Check/install ffmpeg and yt-dlp"
	@echo "  make deps-python Install Python ASR dependencies"
	@echo "  make deps-video-summary Install Python Marlin video summary dependencies"
	@echo "  make run-asr     Run the ASR service"
	@echo "  make run-video-summary Run the Marlin video summary service"
	@echo "  make run         Install deps when needed, then run the Go backend service"
	@echo "  make test        Run Go tests"

deps: check-deps

check-deps: install-ffmpeg install-yt-dlp

deps-python:
	$(PYTHON) -m pip install -r asr_service/requirements.txt

deps-video-summary:
	$(PYTHON) -m pip install -r video_summary_service/requirements.txt

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
	$(PYTHON) -m uvicorn asr_service.app:app --host 0.0.0.0 --port 8001

run-video-summary:
	$(PYTHON) -m uvicorn video_summary_service.app:app --host 0.0.0.0 --port 8002

test:
	$(GO) test ./...
