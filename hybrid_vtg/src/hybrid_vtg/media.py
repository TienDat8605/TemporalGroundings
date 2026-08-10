"""Deterministic timestamp-based video access."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    frame_count: int


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for video decoding") from error
    return cv2


def probe_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(path)
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata for {path}: fps={fps}, frames={frames}")
    return VideoInfo(frames / fps, fps, frames)


def uniform_timestamps(start: float, end: float, count: int) -> tuple[float, ...]:
    if count <= 0 or end <= start:
        return ()
    if count == 1:
        return ((start + end) / 2.0,)
    width = (end - start) / count
    return tuple(min(end - 1e-4, start + (index + 0.5) * width) for index in range(count))


def frame_cache_key(video_path: Path, timestamps: Sequence[float], width: int) -> str:
    payload = f"{video_path.resolve()}\0{width}\0" + ",".join(f"{value:.6f}" for value in timestamps)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def extract_frames(
    video_path: Path,
    timestamps: Sequence[float],
    cache_root: Path,
    *,
    maximum_width: int = 336,
) -> tuple[Path, ...]:
    """Decode exact timestamps once and cache JPEGs under the run results tree."""
    if not timestamps:
        raise ValueError("at least one timestamp is required")
    cv2 = _cv2()
    destination = cache_root / frame_cache_key(video_path, timestamps, maximum_width)
    destination.mkdir(parents=True, exist_ok=True)
    paths = tuple(destination / f"{index:04d}-{timestamp:.3f}.jpg" for index, timestamp in enumerate(timestamps))
    missing = [(timestamp, path) for timestamp, path in zip(timestamps, paths) if not path.is_file()]
    if not missing:
        return paths

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        for timestamp, path in missing:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode {video_path} at {timestamp:.3f}s")
            height, width = frame.shape[:2]
            if width > maximum_width:
                scale = maximum_width / width
                frame = cv2.resize(frame, (maximum_width, max(2, round(height * scale))))
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"cannot write cached frame: {path}")
    finally:
        capture.release()
    return paths
