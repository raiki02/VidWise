GO ?= go

.PHONY: help deps check-deps install-ffmpeg install-yt-dlp run test

help:
	@echo "Usage:"
	@echo "  make deps      Check and install ffmpeg and yt-dlp when missing"
	@echo "  make run       Install deps when needed, then run the backend service"
	@echo "  make test      Run Go tests"

deps: check-deps

check-deps: install-ffmpeg install-yt-dlp

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

test:
	$(GO) test ./...
