"""Video probing and timestamp-driven decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    frames: int


def _decord():
    try:
        import decord
    except ImportError as error:
        raise RuntimeError("decord is required for video access; install SemVID/requirements.txt") from error
    return decord


def probe_video(path: str | Path) -> VideoInfo:
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    reader = _decord().VideoReader(str(video_path), num_threads=1)
    frames = len(reader)
    fps = float(reader.get_avg_fps())
    if frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid video metadata: {video_path}")
    return VideoInfo(duration=frames / fps, fps=fps, frames=frames)


def sample_timestamps(
    duration: float,
    fps: float,
    *,
    start: float = 0.0,
    end: float | None = None,
    max_frames: int = 0,
) -> np.ndarray:
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    upper = duration if end is None else min(duration, float(end))
    lower = max(0.0, float(start))
    if upper <= lower:
        raise ValueError("sampling interval must be non-empty")
    count = max(1, int(np.ceil((upper - lower) * fps)))
    if max_frames > 0:
        count = min(count, max_frames)
    if count == 1:
        return np.asarray([(lower + upper) / 2], dtype=np.float64)
    step = (upper - lower) / count
    return lower + (np.arange(count, dtype=np.float64) + 0.5) * step


def decode_frames(path: str | Path, timestamps: Sequence[float]) -> list[Image.Image]:
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
        raise ValueError("timestamps must be a finite non-empty vector")
    decord = _decord()
    reader = decord.VideoReader(str(path), num_threads=2)
    fps = float(reader.get_avg_fps())
    indices = np.clip(np.floor(times * fps + 0.5).astype(np.int64), 0, len(reader) - 1)
    rgb = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame, mode="RGB") for frame in rgb]


def visual_change(features: np.ndarray) -> np.ndarray:
    vectors = np.asarray(features, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("features must have shape (frames, dimensions)")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    changes = np.zeros(len(normalized), dtype=np.float32)
    if len(normalized) > 1:
        changes[1:] = 1.0 - np.sum(normalized[1:] * normalized[:-1], axis=1)
    return changes
