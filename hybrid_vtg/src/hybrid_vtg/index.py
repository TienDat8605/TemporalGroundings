"""Persistent query-independent coarse video features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coarse_encoder import FrozenSiglipEncoder
from .config import CoarseConfig
from .video import decode_frames, sample_timestamps


@dataclass(frozen=True)
class CoarseIndex:
    video_path: str
    fingerprint: str
    checkpoint: str
    fps: float
    duration: float
    timestamps: np.ndarray
    features: np.ndarray
    decoded_pixels: int = 0
    processor_seconds: float = 0.0
    vision_encoder_seconds: float = 0.0

    def validate(self) -> None:
        if self.timestamps.ndim != 1 or self.features.ndim != 2:
            raise ValueError("invalid coarse index dimensions")
        if not len(self.timestamps) or len(self.timestamps) != len(self.features):
            raise ValueError("coarse timestamps and features must have equal non-zero length")
        if not np.isfinite(self.timestamps).all() or not np.isfinite(self.features).all():
            raise ValueError("coarse index contains non-finite values")


def video_fingerprint(path: str | Path) -> str:
    video = Path(path).resolve()
    stat = video.stat()
    value = f"{video}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode()).hexdigest()


def cache_path(cache_dir: Path, path: str | Path, config: CoarseConfig) -> Path:
    payload = json.dumps({
        "video": video_fingerprint(path),
        "checkpoint": config.checkpoint,
        "fps": config.fps,
        "max_frames": config.max_frames,
    }, sort_keys=True)
    return cache_dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.npz"


def save_index(path: Path, index: CoarseIndex) -> None:
    index.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({
        "video_path": index.video_path,
        "fingerprint": index.fingerprint,
        "checkpoint": index.checkpoint,
        "fps": index.fps,
        "duration": index.duration,
        "decoded_pixels": index.decoded_pixels,
        "processor_seconds": index.processor_seconds,
        "vision_encoder_seconds": index.vision_encoder_seconds,
    }, sort_keys=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata=metadata, timestamps=index.timestamps, features=index.features)
    temporary.replace(path)


def load_index(path: Path) -> CoarseIndex:
    with np.load(path, allow_pickle=False) as value:
        index = CoarseIndex(
            **json.loads(str(value["metadata"])),
            timestamps=np.asarray(value["timestamps"], dtype=np.float64),
            features=np.asarray(value["features"], dtype=np.float32),
        )
    index.validate()
    return index


def build_index(
    video_path: str,
    duration: float,
    config: CoarseConfig,
    encoder: FrozenSiglipEncoder,
) -> CoarseIndex:
    timestamps = sample_timestamps(duration, config.fps, max_frames=config.max_frames)
    frames = decode_frames(video_path, timestamps)
    decoded_pixels = sum(image.width * image.height for image in frames)
    features = encoder.encode_images(frames)
    stats = encoder.last_encode_stats
    return CoarseIndex(
        video_path=video_path,
        fingerprint=video_fingerprint(video_path),
        checkpoint=config.checkpoint,
        fps=len(timestamps) / duration,
        duration=duration,
        timestamps=timestamps,
        features=features,
        decoded_pixels=decoded_pixels,
        processor_seconds=float(stats.get("processor_seconds", 0.0)),
        vision_encoder_seconds=float(stats.get("vision_encoder_seconds", 0.0)),
    )
