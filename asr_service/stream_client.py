import argparse
import asyncio
import json
import wave

import websockets


async def stream_wav(ws_url: str, wav_path: str, chunk_ms: int, language: str) -> None:
    async with websockets.connect(ws_url, max_size=2**24) as websocket:
        await websocket.send(json.dumps({"event": "config", "language": language}))

        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            if channels != 1 or sample_rate != 16000 or sample_width != 2:
                raise ValueError("WAV must be 16 kHz mono, 16-bit PCM")

            frames_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    break
                await websocket.send(data)
                await asyncio.sleep(chunk_ms / 1000.0)

        await websocket.send(json.dumps({"event": "end"}))

        try:
            async for message in websocket:
                print(message)
        except websockets.ConnectionClosed:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a WAV file to the ASR WebSocket endpoint.")
    parser.add_argument("--ws-url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8001/stream")
    parser.add_argument("--wav", required=True, help="Path to 16 kHz mono PCM WAV file")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Chunk size in milliseconds")
    parser.add_argument("--language", default="zh", help="Language code")
    args = parser.parse_args()

    asyncio.run(stream_wav(args.ws_url, args.wav, args.chunk_ms, args.language))


if __name__ == "__main__":
    main()

