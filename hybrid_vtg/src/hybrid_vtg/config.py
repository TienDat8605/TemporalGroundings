"""Configuration for full-video spatial-token benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SPATIAL_POLICIES = (
    "dense", "semvid", "uniform", "tpsa_query", "tpsa_motion", "tpsa_boundary",
)
OBSERVATION_POLICIES = ("single_pass", "hmve")


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
    model_gpu_memory_ratio: float = 0.95

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
        if not 0 < self.model_gpu_memory_ratio < 1:
            raise ValueError("model GPU memory ratio must be in (0, 1)")


@dataclass(frozen=True)
class SpatialAllocatorConfig:
    spatial_policy: str = "semvid"
    retention_ratio: float = 0.125
    mmr_lambda: float = 0.9
    relevance_top_fraction: float = 0.10
    auxiliary_fraction: float = 0.10
    boundary_share: float = 0.50
    motion_query_beta: float = 0.50
    evidence_mad_multiplier: float = 2.0
    boundary_window_seconds: float = 1.0
    boundary_nms_seconds: float = 4.0
    boundary_expansion_seconds: float = 1.0
    maximum_boundary_bands: int = 4

    def __post_init__(self) -> None:
        if self.spatial_policy not in SPATIAL_POLICIES:
            raise ValueError(f"spatial policy must be one of {SPATIAL_POLICIES}")
        if not 0 <= self.retention_ratio <= 1:
            raise ValueError("retention ratio must be in [0, 1]")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("MMR lambda must be in [0, 1]")
        if not 0 < self.relevance_top_fraction <= 1:
            raise ValueError("relevance top fraction must be in (0, 1]")
        if not 0 <= self.auxiliary_fraction <= 0.10:
            raise ValueError("auxiliary fraction must be in [0, 0.10]")
        if not 0 <= self.boundary_share <= 1:
            raise ValueError("boundary share must be in [0, 1]")
        if not 0 <= self.motion_query_beta <= 1:
            raise ValueError("motion/query beta must be in [0, 1]")
        if self.evidence_mad_multiplier < 0:
            raise ValueError("evidence MAD multiplier must be non-negative")
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

    def allocator_constants(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationConfig:
    policy: str = "single_pass"
    scout_fps: float = 0.5
    scout_total_pixel_tokens: int = 2048
    maximum_corridors: int = 4
    corridor_margin_seconds: float = 4.0
    minimum_corridor_seconds: float = 8.0
    corridor_nms_seconds: float = 8.0
    detailed_budget_fraction: float = 0.80
    deduplication_similarity: float = 0.98

    def __post_init__(self) -> None:
        if self.policy not in OBSERVATION_POLICIES:
            raise ValueError(f"observation policy must be one of {OBSERVATION_POLICIES}")
        if self.scout_fps <= 0 or self.scout_total_pixel_tokens <= 0:
            raise ValueError("HMVE scout FPS and pixel-token budget must be positive")
        if self.maximum_corridors <= 0:
            raise ValueError("HMVE must retain at least one candidate corridor")
        if min(
            self.corridor_margin_seconds,
            self.minimum_corridor_seconds,
            self.corridor_nms_seconds,
        ) < 0:
            raise ValueError("HMVE corridor time constants must be non-negative")
        if self.minimum_corridor_seconds == 0:
            raise ValueError("HMVE minimum corridor duration must be positive")
        if not 0 <= self.detailed_budget_fraction <= 1:
            raise ValueError("HMVE detailed budget fraction must be in [0, 1]")
        if not -1 <= self.deduplication_similarity <= 1:
            raise ValueError("HMVE deduplication similarity must be in [-1, 1]")


@dataclass(frozen=True)
class PipelineConfig:
    grounder: GrounderConfig = field(default_factory=GrounderConfig)
    spatial_allocator: SpatialAllocatorConfig = field(default_factory=SpatialAllocatorConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
