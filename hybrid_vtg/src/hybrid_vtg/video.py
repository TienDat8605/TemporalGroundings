"""Video probing and timestamp-driven decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


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
