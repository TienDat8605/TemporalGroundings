"""One-call, global-anchor temporal grounding under a 64-frame budget."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample, TemporalEvidence
from ...media import uniform_timestamps
from ..coarse_to_fine_64 import (
    EmbeddingRouter,
    Window,
    cached_content_windows,
    coalesce_windows,
    retained_window_count,
    scene_cache_path,
    select_temporally_diverse_windows,
)

FRAME_BUDGET = 64
MAX_ROUTED_WINDOWS = 16
ROUTER_FRAMES_PER_WINDOW = 4
ROUTING_MARGIN = 0.5
CORRIDOR_CONTEXT_SECONDS = 4.0
CORRIDOR_MAX_SECONDS = 64.0


@dataclass(frozen=True)
class Observation:
    timestamp: float
    role: str
    region_id: int | None = None


@dataclass(frozen=True)
class RouteDecision:
    confident: bool
    reasons: tuple[str, ...]
    robust_margin: float
    raw_top: int
    aggregate_top: int
    whole_window_top: int
    occurrence_top: int


def _top_index(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("at least one routing score is required")
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


def route_confidence(
    scores: Sequence[float],
    router_telemetry: dict[str, Any],
    *,
    margin_threshold: float = ROUTING_MARGIN,
) -> RouteDecision:
    """Require semantic-view, visual-view, and robust-margin agreement."""
    if len(scores) == 1:
        return RouteDecision(True, (), math.inf, 0, 0, 0, 0)
    views = router_telemetry.get("query_view_scores") or {"raw": list(scores)}
    raw_scores = [float(value) for value in views.get("raw", scores)]
    whole = [float(value) for value in router_telemetry.get("window_similarity", raw_scores)]
    occurrence = [
        float(value)
        for value in router_telemetry.get("frame_occurrence_similarity", raw_scores)
    ]
    raw_top = _top_index(raw_scores)
    aggregate_top = _top_index(scores)
    whole_top = _top_index(whole)
    occurrence_top = _top_index(occurrence)
    values = np.asarray(scores, dtype=np.float64)
    ranked = np.sort(values)[::-1]
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    robust_margin = float((ranked[0] - ranked[1]) / (spread + 1e-6))
    reasons = []
    if raw_top != aggregate_top:
        reasons.append("query-view-disagreement")
    if whole_top != occurrence_top:
        reasons.append("visual-view-disagreement")
    if robust_margin < margin_threshold:
        reasons.append("low-robust-margin")
    return RouteDecision(
        not reasons,
        tuple(reasons),
        robust_margin,
        raw_top,
        aggregate_top,
        whole_top,
        occurrence_top,
    )


def corridor_for_window(
    window: Window,
    duration: float,
    *,
    context_seconds: float = CORRIDOR_CONTEXT_SECONDS,
    maximum_seconds: float = CORRIDOR_MAX_SECONDS,
) -> Window:
    start = max(0.0, window.start - context_seconds)
    end = min(duration, window.end + context_seconds)
    if end - start <= maximum_seconds:
        return Window(start, end)
    center = (window.start + window.end) / 2.0
    start = max(0.0, min(duration - maximum_seconds, center - maximum_seconds / 2.0))
    return Window(start, min(duration, start + maximum_seconds))


def _add_unique(
    output: dict[float, Observation],
    observation: Observation,
) -> None:
    key = round(observation.timestamp, 9)
    existing = output.get(key)
    if existing is None or observation.role == "global_anchor":
        output[key] = observation


def plan_anchored_evidence(
    windows: Sequence[Window],
    corridors: Sequence[Window],
    duration: float,
    *,
    budget: int = FRAME_BUDGET,
) -> tuple[Observation, ...]:
    """Keep one timeline anchor per routed region and spend the rest locally."""
    if duration <= 0 or budget <= 0:
        raise ValueError("duration and frame budget must be positive")
    if len(windows) > budget:
        raise ValueError("the frame budget cannot protect every routed window")
    if not corridors:
        return tuple(
            Observation(timestamp, "global_anchor")
            for timestamp in uniform_timestamps(0.0, duration, budget)
        )

    observations: dict[float, Observation] = {}
    for index, window in enumerate(windows):
        _add_unique(
            observations,
            Observation((window.start + window.end) / 2.0, "global_anchor", index),
        )

    remaining = budget - len(observations)
    base, remainder = divmod(remaining, len(corridors))
    for region_id, corridor in enumerate(corridors):
        count = base + int(region_id < remainder)
        for timestamp in uniform_timestamps(corridor.start, corridor.end, count):
            _add_unique(observations, Observation(timestamp, "corridor", region_id))

    # An anchor can coincide with a dense observation. Refill deterministically from
    # a fine full-video grid so the grounding budget remains exact.
    if len(observations) < budget:
        for timestamp in uniform_timestamps(0.0, duration, budget * 8):
            _add_unique(observations, Observation(timestamp, "exploration"))
            if len(observations) == budget:
                break
    if len(observations) != budget:
        raise RuntimeError(f"planned {len(observations)} observations for a {budget}-frame budget")
    return tuple(sorted(observations.values(), key=lambda value: value.timestamp))


def assign_observation_roles(
    evidence: TemporalEvidence,
    observations: Sequence[Observation],
) -> None:
    """Map encoded temporal units back to their nearest planned observation role."""
    times = np.asarray([value.timestamp for value in observations], dtype=np.float64)
    roles = []
    for timestamp in evidence.timestamps:
        index = int(np.argmin(np.abs(times - float(timestamp))))
        roles.append(observations[index].role)
    evidence.roles = tuple(roles)
    evidence.metadata["observation_plan"] = [
        {
            "timestamp": value.timestamp,
            "role": value.role,
            "region_id": value.region_id,
        }
        for value in observations
    ]


def _decoded_pixels(evidence: TemporalEvidence) -> int | None:
    paths = evidence.metadata.get("frame_paths")
    if not isinstance(paths, list) or not paths:
        return None
    from PIL import Image

    total = 0
    for path in paths:
        with Image.open(path) as image:
            total += image.width * image.height
    return total


class AnchoredCorridor64(Method):
    """Route semantically, retain global anchors, and ground exactly once."""

    name = "anchored-corridor-64"

    def __init__(self, router: EmbeddingRouter | None = None) -> None:
        self.router = router or EmbeddingRouter()
        self._prepare_root: Path | None = None

    @staticmethod
    def _routed_windows(windows: Sequence[Window]) -> list[Window]:
        return coalesce_windows(windows, min(len(windows), MAX_ROUTED_WINDOWS))

    def prepare(self, samples: Sequence[Sample], cache_root: Path) -> None:
        self._prepare_root = cache_root
        prepared: set[Path] = set()
        for sample in samples:
            path = scene_cache_path(sample, cache_root)
            if path not in prepared:
                cached_content_windows(sample, cache_root)
                prepared.add(path)
        jobs = []
        for sample in samples:
            windows, _ = cached_content_windows(sample, cache_root)
            if len(windows) > 1:
                jobs.append((sample, self._routed_windows(windows)))
        if not jobs or not isinstance(self.router, EmbeddingRouter):
            return

        import torch

        self.router.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.router.cache_query_views([sample.query for sample, _ in jobs], cache_root)
            unique_visual_jobs: dict[Path, tuple[Sample, list[Window]]] = {}
            for sample, windows in jobs:
                path, _ = self.router._video_cache_path(
                    sample,
                    windows,
                    ROUTER_FRAMES_PER_WINDOW,
                    cache_root,
                )
                unique_visual_jobs.setdefault(path, (sample, windows))
            for sample, windows in tqdm(
                unique_visual_jobs.values(),
                desc=f"anchored visual cache ({self.router.device})",
                unit="video",
            ):
                self.router.rank(
                    sample,
                    windows,
                    ROUTER_FRAMES_PER_WINDOW,
                    cache_root,
                )
        finally:
            self.router.unload(fallback_device="cpu")

    def _cached_windows(self, sample: Sample, cache_dir: Path) -> tuple[list[Window], str]:
        return cached_content_windows(sample, self._prepare_root or cache_dir)

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        self.validate_model(model)
        if getattr(model, "encoder_pruning", "none") != "none" or getattr(
            model, "post_pruning", "none"
        ) != "none":
            raise ValueError(
                "anchored-corridor-64 currently requires dense evidence; evaluate Mage and "
                "SemVID as separate follow-up ablations"
            )

        windows, source = self._cached_windows(sample, cache_dir)
        routed = self._routed_windows(windows)
        router_started = perf_counter()
        if len(routed) == 1:
            scores = [1.0]
            decision = RouteDecision(True, (), math.inf, 0, 0, 0, 0)
            selected = [0]
            corridors = [Window(0.0, sample.duration)]
            router_details: dict[str, Any] = {}
            route_mode = "single-window-bypass"
        else:
            scores = self.router.rank_multigranular(
                sample,
                routed,
                ROUTER_FRAMES_PER_WINDOW,
                self._prepare_root or cache_dir,
            )
            router_details = dict(getattr(self.router, "last_telemetry", {}))
            decision = route_confidence(scores, router_details)
            if decision.confident:
                count = 1 if sample.cardinality == "single" else retained_window_count(len(routed))
                selected, _ = select_temporally_diverse_windows(routed, scores, count)
                corridors = [
                    corridor_for_window(routed[index], sample.duration) for index in selected
                ]
                route_mode = "corridor"
            else:
                selected = []
                corridors = []
                route_mode = "full-video-fallback"
        router_seconds = perf_counter() - router_started

        observations = plan_anchored_evidence(routed, corridors, sample.duration)
        timestamps = tuple(value.timestamp for value in observations)
        encode_started = perf_counter()
        evidence = model.encode(sample, timestamps)
        encode_seconds = perf_counter() - encode_started
        assign_observation_roles(evidence, observations)
        predict_started = perf_counter()
        result = model.predict(
            sample,
            evidence,
            GroundingContext(0.0, sample.duration),
        )
        predict_seconds = perf_counter() - predict_started

        role_counts = {
            role: sum(value.role == role for value in observations)
            for role in ("global_anchor", "corridor", "exploration")
        }
        decoded_pixels = _decoded_pixels(evidence)
        dense_tokens = evidence.metadata.get("dense_evidence_units", evidence.size)
        retained_tokens = evidence.metadata.get("encoder_retained_evidence_units", evidence.size)
        resource_ledger = {
            "router_index": {
                "source_frames": (
                    0 if len(routed) == 1 else len(routed) * ROUTER_FRAMES_PER_WINDOW
                ),
                "video_embedding_cache_hit": router_details.get(
                    "video_embedding_cache_hit"
                ),
            },
            "grounding": {
                "source_frames": len(observations),
                "decoded_pixels": decoded_pixels,
                "dense_visual_tokens": dense_tokens,
                "retained_visual_tokens": retained_tokens,
                "encoder_calls": 1,
                "primary_grounder_calls": 1,
            },
        }
        return Prediction(
            result.spans,
            result.raw_output,
            {
                **result.telemetry,
                "method": self.name,
                "window_source": source,
                "route_mode": route_mode,
                "route_confident": decision.confident,
                "route_reasons": list(decision.reasons),
                "routing_margin_threshold": ROUTING_MARGIN,
                "robust_margin": decision.robust_margin,
                "route_agreement": {
                    "raw_top": decision.raw_top,
                    "aggregate_top": decision.aggregate_top,
                    "whole_window_top": decision.whole_window_top,
                    "occurrence_top": decision.occurrence_top,
                },
                "windows": [{"start": value.start, "end": value.end} for value in routed],
                "scores": scores,
                "selected": selected,
                "corridors": [
                    {"start": value.start, "end": value.end} for value in corridors
                ],
                "router_cache": router_details,
                "router_seconds": router_seconds,
                "router_index_frames": (
                    0 if len(routed) == 1 else len(routed) * ROUTER_FRAMES_PER_WINDOW
                ),
                "grounder_frames": len(observations),
                "total_frames": len(observations),
                "frame_budget_scope": "grounding evidence; cached router index reported separately",
                "encoder_calls": 1,
                "primary_grounder_calls": 1,
                "observation_role_counts": role_counts,
                "observation_plan": [
                    {
                        "timestamp": value.timestamp,
                        "role": value.role,
                        "region_id": value.region_id,
                    }
                    for value in observations
                ],
                "encoder_dense_tokens": dense_tokens,
                "encoder_retained_tokens": retained_tokens,
                "resource_ledger": resource_ledger,
                "timing": {
                    "encode_seconds": encode_seconds,
                    "predict_seconds": predict_seconds,
                    "total_seconds": encode_seconds + predict_seconds,
                },
            },
        )


__all__ = [
    "AnchoredCorridor64",
    "Observation",
    "RouteDecision",
    "assign_observation_roles",
    "corridor_for_window",
    "plan_anchored_evidence",
    "route_confidence",
]
