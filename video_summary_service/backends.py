import logging
import os
import re
from typing import Any, List, Optional, Protocol, Tuple

from typing_extensions import TypedDict

logger = logging.getLogger("vidwise.video_summary.backend")


# ---------------------------------------------------------------------------
# Constants — must match exactly what the model was fine-tuned on
# ---------------------------------------------------------------------------

CAPTION_PROMPT: str = (
    "Provide a spatial description of this clip followed by time-ranged events.\n"
    "For each event, give the time range as <start - end> and a short description."
)

GROUNDING_PROMPT_TEMPLATE: str = (
    'Identify the timestamps during which "{event}" takes place. '
    'Output the time range as "From <start> to <end>." (numbers in seconds).'
)


# ---------------------------------------------------------------------------
# Thinking-tag stripping
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_PREFIX = re.compile(r"^\s*<think>\s*\n*", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>\s*", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    out = _THINK_BLOCK.sub("", text)
    out = _THINK_PREFIX.sub("", out)
    out = _THINK_CLOSE.sub("", out)
    return out.strip()


# ---------------------------------------------------------------------------
# Mode 1 — dense caption parser
# ---------------------------------------------------------------------------

class Event(TypedDict):
    start: float
    end: float
    description: str


_EVENT_LINE = re.compile(
    r"^\s*<?\s*(\d+\.?\d*)\s*(?:seconds?|secs?|s)?\s*-\s*"
    r"(\d+\.?\d*)\s*(?:seconds?|secs?|s)?\s*>?\s*[:\-]?\s*(.+?)\s*$"
)


def _parse_events(events_block: str) -> List[Event]:
    out: List[Event] = []
    for raw_line in events_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _EVENT_LINE.match(line)
        if not m:
            continue
        start = float(m.group(1))
        end = float(m.group(2))
        desc = m.group(3).strip().lstrip("-").strip()
        if end <= start or not desc:
            continue
        out.append(Event(start=start, end=end, description=desc))
    return out


def parse_caption(text: str) -> Tuple[str, str, List[Event]]:
    cleaned = strip_thinking(text)

    scene_match = re.search(
        r"(?:^|\n)\s*Scene\s*:\s*(.*?)(?=\n\s*Events\s*:|\Z)",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    events_match = re.search(
        r"(?:^|\n)\s*Events\s*:\s*(.*)\Z",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )

    if scene_match:
        scene = scene_match.group(1).strip()
    else:
        scene_lines: List[str] = []
        for line in cleaned.splitlines():
            if _EVENT_LINE.match(line.strip()):
                break
            scene_lines.append(line)
        scene = "\n".join(scene_lines).strip()

    events_block = events_match.group(1) if events_match else cleaned
    events = _parse_events(events_block)

    return cleaned, scene, events


# ---------------------------------------------------------------------------
# Mode 2 — temporal grounding parser
# ---------------------------------------------------------------------------

_SPAN_RE = re.compile(
    r"From\s+(\d+\.?\d*)\s*(?:s|sec)?\s+to\s+(\d+\.?\d*)\s*(?:s|sec)?\.?",
    re.IGNORECASE,
)


def parse_span(text: str) -> Tuple[str, Optional[Tuple[float, float]]]:
    cleaned = strip_thinking(text)
    m = _SPAN_RE.search(cleaned)
    if not m:
        return cleaned, None
    start = float(m.group(1))
    end = float(m.group(2))
    if end <= start:
        return cleaned, None
    return cleaned, (start, end)


# ---------------------------------------------------------------------------
# Result dicts
# ---------------------------------------------------------------------------

class CaptionResult(TypedDict):
    caption: str
    scene: str
    events: List[Event]
    raw: str


class FindResult(TypedDict):
    raw: str
    span: Optional[Tuple[float, float]]
    format_ok: bool


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class VideoSummaryBackend(Protocol):
    def caption(
        self,
        video_path: str,
        *,
        max_new_tokens: int = 2048,
        prompt: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> CaptionResult: ...

    def find(
        self,
        video_path: str,
        event: str,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> FindResult: ...


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

class HuggingFaceBackend:
    def __init__(self, model_config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": _torch_dtype(model_config["dtype"]),
        }
        device = str(model_config["device"]).strip().lower()
        if device == "cuda":
            kwargs["device_map"] = {"": "cuda"}
        elif device == "cpu":
            kwargs["device_map"] = {"": "cpu"}
        else:
            kwargs["device_map"] = "auto"

        logger.info(
            "loading HuggingFace model model=%s device=%s",
            model_config["name"],
            model_config["device"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_config["name"], **kwargs)
        if model_config.get("compile"):
            logger.info("compiling HuggingFace model")
            self.model.compile()
        logger.info("HuggingFace model loaded")

    def caption(
        self,
        video_path: str,
        *,
        max_new_tokens: int = 2048,
        prompt: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> CaptionResult:
        result = self.model.caption(
            video_path,
            max_new_tokens=max_new_tokens,
            prompt=prompt or None,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return CaptionResult(
            caption=result.get("caption", ""),
            scene=result.get("scene", ""),
            events=result.get("events", []),
            raw=result.get("raw", ""),
        )

    def find(
        self,
        video_path: str,
        event: str,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> FindResult:
        result = self.model.find(
            video_path,
            event=event,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        span = result.get("span")
        return FindResult(
            raw=result.get("raw", ""),
            span=tuple(span) if span is not None else None,
            format_ok=bool(result.get("format_ok", False)),
        )


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

class MLXBackend:
    def __init__(self, model_config: dict[str, Any]) -> None:
        from mlx_vlm import load

        model_name = model_config["name"]
        logger.info("loading MLX model model=%s", model_name)
        self.model, self.processor = load(model_name, trust_remote_code=True)
        self.config = _load_mlx_config(model_name)
        logger.info("MLX model loaded")

    def caption(
        self,
        video_path: str,
        *,
        max_new_tokens: int = 2048,
        prompt: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> CaptionResult:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt_text = prompt or CAPTION_PROMPT
        frames = _extract_video_frames(video_path)

        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt_text,
            num_images=len(frames),
        )

        raw = generate(
            self.model,
            self.processor,
            formatted_prompt,
            frames,
            max_tokens=max_new_tokens,
            temp=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
        )

        cleaned, scene, events = parse_caption(raw)
        return CaptionResult(caption=cleaned, scene=scene, events=events, raw=raw)

    def find(
        self,
        video_path: str,
        event: str,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> FindResult:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt_text = GROUNDING_PROMPT_TEMPLATE.format(event=event)
        frames = _extract_video_frames(video_path)

        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt_text,
            num_images=len(frames),
        )

        raw = generate(
            self.model,
            self.processor,
            formatted_prompt,
            frames,
            max_tokens=max_new_tokens,
            temp=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
        )

        cleaned, span = parse_span(raw)
        return FindResult(raw=cleaned, span=span, format_ok=span is not None)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(model_config: dict[str, Any]) -> VideoSummaryBackend:
    provider = str(model_config.get("provider") or "huggingface").strip().lower()
    if provider in {"mlx", "mlx-vlm"}:
        return MLXBackend(model_config)
    if provider in {"huggingface", "hf", "transformers"}:
        return HuggingFaceBackend(model_config)
    raise ValueError(
        f"video_summary.model.provider must be 'huggingface' or 'mlx', got: {provider!r}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_dtype(name: str) -> Any:
    import torch

    normalized = str(name).strip().lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    return torch.bfloat16


def _extract_video_frames(video_path: str, target_fps: float = 2.0) -> list[Any]:
    """Extract video frames as PIL Images at the given target FPS."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    frame_interval = max(1, int(video_fps / target_fps))

    frames: list[Image.Image] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            frame_idx += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"no frames extracted from video: {video_path}")
    return frames


def _load_mlx_config(model_name: str) -> dict[str, Any]:
    from mlx_vlm.utils import load_config

    return load_config(model_name)
