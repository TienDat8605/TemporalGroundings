"""Configuration for full-video spatial-token benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SPATIAL_POLICIES = (
    "dense", "semvid", "uniform", "tpsa_query", "tpsa_motion", "tpsa_boundary",
)


@dataclass(frozen=True)
class GrounderConfig:
    model: str = "Qwen/Qwen3-VL-4B-Thinking"
    fps: float = 2.0
    max_frames: int = 768
    max_new_tokens: int = 512
    force_stop_thinking: bool = True
    total_pixel_tokens: int = 16384
    minimum_pixel_tokens: int = 16
    dtype: str = "auto"
    attention: str = "sdpa"
    batch_size: int = 1
    pairing_lookahead: int = 16
    preprocess_workers: int = 0
    prefetch_depth: int = 0
    pinned_memory_limit_bytes: int = 4 * 1024**3
    capture_validation_logits: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.max_frames <= 0 or self.max_new_tokens <= 0:
            raise ValueError("grounder FPS and frame/token limits must be positive")
        if self.batch_size not in {1, 2}:
            raise ValueError("verified Qwen batch size must be 1 or 2")
        if self.pairing_lookahead <= 0:
            raise ValueError("Qwen pairing lookahead must be positive")
        if self.preprocess_workers not in {0, 1}:
            raise ValueError("Qwen preprocessing supports zero or one worker")
        if self.prefetch_depth < 0:
            raise ValueError("Qwen prefetch depth must be non-negative")
        if self.preprocess_workers == 0 and self.prefetch_depth:
            raise ValueError("Qwen prefetch requires one preprocessing worker")
        if self.pinned_memory_limit_bytes <= 0:
            raise ValueError("pinned-memory limit must be positive")


@dataclass(frozen=True)
class SpatialAllocatorConfig:
    spatial_policy: str = "semvid"
    retention_ratio: float = 0.125
    mmr_lambda: float = 0.9
    relevance_top_fraction: float = 0.10
    boundary_quota_fraction: float = 0.10
    motion_neighborhood_radius: int = 2
    boundary_window_seconds: float = 1.0
    boundary_nms_seconds: float = 4.0
    boundary_expansion_seconds: float = 1.0
    maximum_boundary_bands: int = 4
    query_core_fraction: float = 0.80
    motion_bonus_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.spatial_policy not in SPATIAL_POLICIES:
            raise ValueError(f"spatial policy must be one of {SPATIAL_POLICIES}")
        if not 0 <= self.retention_ratio <= 1:
            raise ValueError("retention ratio must be in [0, 1]")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("MMR lambda must be in [0, 1]")
        if not 0 < self.relevance_top_fraction <= 1:
            raise ValueError("relevance top fraction must be in (0, 1]")
        if not 0 <= self.boundary_quota_fraction <= 0.5:
            raise ValueError("boundary quota fraction must be in [0, 0.5]")
        if self.motion_neighborhood_radius < 0:
            raise ValueError("motion neighborhood radius must be non-negative")
        if min(
            self.boundary_window_seconds,
            self.boundary_nms_seconds,
            self.boundary_expansion_seconds,
        ) < 0:
            raise ValueError("boundary time constants must be non-negative")
        if self.boundary_window_seconds == 0:
            raise ValueError("boundary window must be positive")
        if self.maximum_boundary_bands <= 0:
            raise ValueError("maximum boundary bands must be positive")
        if not 0 <= self.query_core_fraction <= 1:
            raise ValueError("query core fraction must be in [0, 1]")
        if not 0 <= self.motion_bonus_fraction <= 1:
            raise ValueError("motion bonus fraction must be in [0, 1]")
        if self.query_core_fraction + 2 * self.boundary_quota_fraction > 1 + 1e-8:
            raise ValueError("query core and two boundary quotas must not exceed the token budget")

    def allocator_constants(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    grounder: GrounderConfig = field(default_factory=GrounderConfig)
    spatial_allocator: SpatialAllocatorConfig = field(default_factory=SpatialAllocatorConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
