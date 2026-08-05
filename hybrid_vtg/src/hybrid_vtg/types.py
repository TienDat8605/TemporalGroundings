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

    def validate(self) -> None:
        if not self.id or not self.video_path or not self.query.strip():
            raise ValueError("sample id, video path, and query are required")
        if self.duration <= 0:
            raise ValueError(f"sample {self.id!r} has non-positive duration")
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
class TemporalRoute:
    components: tuple[Component, ...]
    selected_candidates: tuple[int, ...]
    confidence_margin: float
    low_confidence_fallback: bool
    retained_union_seconds: float


@dataclass(frozen=True)
class GroundingPrediction:
    interval: tuple[float, float]
    component: Component
    raw_text: str
    semvid_stats: dict[str, Any]
    token_roles: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["interval"] = list(self.interval)
        return value
