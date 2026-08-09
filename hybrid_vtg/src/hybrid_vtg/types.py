"""Shared immutable records and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Sample:
    id: str
    video: str
    video_path: str
    duration: float
    query: str
    targets: tuple[tuple[float, float], ...] = ()
    group: str = "custom"
    cardinality: str = "single"

    def validate(self) -> None:
        if not self.id or not self.video_path or not self.query.strip():
            raise ValueError("sample id, video path, and query are required")
        if self.duration <= 0:
            raise ValueError(f"sample {self.id!r} has non-positive duration")
        if self.cardinality not in {"single", "multi"}:
            raise ValueError(f"sample {self.id!r} has invalid cardinality {self.cardinality!r}")
        for start, end in self.targets:
            if not 0 <= start < end <= self.duration + 1e-3:
                raise ValueError(f"sample {self.id!r} has invalid target {(start, end)}")


@dataclass(frozen=True)
class Component:
    start: float
    end: float
    score: float
    source_candidates: tuple[int, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class GroundingPrediction:
    interval: tuple[float, float] | None
    component: Component
    raw_text: str
    spatial_stats: dict[str, Any]
    token_roles: dict[str, int]
    telemetry: dict[str, Any]
    intervals: tuple[tuple[float, float], ...] = ()

    @property
    def semvid_stats(self) -> dict[str, Any]:
        """Deprecated upstream-compatible alias for ``spatial_stats``."""
        return self.spatial_stats

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["interval"] = list(self.interval) if self.interval is not None else None
        value["intervals"] = [list(interval) for interval in self.intervals]
        value["semvid_stats"] = value["spatial_stats"]
        return value
