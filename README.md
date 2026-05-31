# video-extractor

视频、音频和字幕文本提取服务。

## Architecture

- Go/Gin exposes the public HTTP API.
- Go keeps `/extract` unchanged and returns a video, audio, or text file based on `type`.
- Python exposes the faster-whisper ASR model through HTTP.
- Go wraps the Python ASR endpoint as an Eino tool and uses it in the text extraction path.

## Setup

Copy and edit the config:

```bash
cp config.example.yaml config.yaml
```

Install command-line dependencies:

```bash
make deps
```

Install Python ASR dependencies:

```bash
make deps-python
```

## Run

Start the ASR service:

```bash
make run-asr
```

Start the Go backend in another terminal:

```bash
make run
```

## API

The HTTP API remains:

```bash
curl -G 'http://localhost:8080/extract' \
  --data-urlencode 'url=https://example.com/video' \
  --data-urlencode 'name=demo' \
  --data-urlencode 'type=text' \
  -o demo.txt
```

Supported `type` values:

- `video`: returns a device-playable mp4
- `audio`: returns an mp3
- `text` or `transcript`: returns formatted transcript text

## Web UI

Start the server, then open `http://localhost:8080/` in a browser to use the built-in extraction form.
