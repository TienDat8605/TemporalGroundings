"""Pipeline configuration with portable, benchmark-independent defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CoarseConfig:
    enabled: bool = True
    checkpoint: str = "google/siglip2-base-patch16-224"
    fps: float = 0.5
    batch_size: int = 32
    max_frames: int = 2048
    scales: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0)
    stride_ratio: float = 0.5
    mean_weight: float = 0.5
    union_budget_seconds: float = 120.0
    maximum_candidates: int = 8
    nms_iou: float = 0.7
    minimum_uncovered_seconds: float = 1.0
    halo_seconds: float = 2.0
    low_confidence_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.batch_size <= 0 or self.max_frames <= 0:
            raise ValueError("coarse FPS, batch size, and frame cap must be positive")
        if not self.scales or any(value <= 0 for value in self.scales):
            raise ValueError("at least one positive temporal scale is required")
        if self.union_budget_seconds <= 0 or self.maximum_candidates <= 0:
            raise ValueError("temporal budgets must be positive")
        if not 0 <= self.mean_weight <= 1 or not 0 <= self.nms_iou <= 1:
            raise ValueError("coarse score and NMS weights must be in [0, 1]")


@dataclass(frozen=True)
class SemVIDConfig:
    enabled: bool = True
    model: str = "Qwen/Qwen3-VL-4B-Thinking"
    fps: float = 2.0
    retention_ratio: float = 0.125
    object_ratio: float = 0.6
    mmr_lambda: float = 0.9
    frame_weight_alpha: float = 0.7
    motion_query_beta: float = 0.5
    max_frames: int = 768
    max_new_tokens: int = 200
    force_stop_thinking: bool = True
    total_pixel_tokens: int = 16384
    minimum_pixel_tokens: int = 16
    dtype: str = "auto"
    attention: str = "sdpa"
    timestamp_mode: str = "absolute"

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.max_frames <= 0 or self.max_new_tokens <= 0:
            raise ValueError("expert FPS and frame/token limits must be positive")
        for name, value in (
            ("retention_ratio", self.retention_ratio), ("object_ratio", self.object_ratio),
            ("mmr_lambda", self.mmr_lambda), ("frame_weight_alpha", self.frame_weight_alpha),
            ("motion_query_beta", self.motion_query_beta),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.timestamp_mode not in {"absolute", "relative", "auto"}:
            raise ValueError("timestamp mode must be absolute, relative, or auto")


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    fps: float = 8.0
    radius_seconds: float = 2.0
    evidence_window_seconds: float = 0.5
    continuity_weight: float = 0.25
    minimum_gain: float = 0.01

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.radius_seconds <= 0 or self.evidence_window_seconds <= 0:
            raise ValueError("refinement FPS, radius, and evidence window must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    coarse: CoarseConfig = field(default_factory=CoarseConfig)
    semvid: SemVIDConfig = field(default_factory=SemVIDConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
