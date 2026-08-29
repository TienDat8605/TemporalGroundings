"""Evidence planning and frame budget allocation for SGDE (Idea 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ...contracts import GroundingContext, TemporalEvidence
from ...media import uniform_timestamps
from .proposals import CandidateProposal

FRAME_BUDGET = 64
DEFAULT_NUM_ANCHORS = 16
DEFAULT_CONTEXT_SECONDS = 6.0


@dataclass(frozen=True)
class Observation:
    timestamp: float
    role: str
    candidate_id: int | None = None

    def to_dict(self) -> dict[str, float | str | int | None]:
        return {
            "timestamp": round(self.timestamp, 4),
            "role": self.role,
            "candidate_id": self.candidate_id,
        }


def plan_sgde_windows(
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    total_budget: int = FRAME_BUDGET,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    min_window: float = 24.0,
) -> list[tuple[GroundingContext, int]]:
    """Cluster candidates into adaptive 1-2 windows using logarithmic margin, sqrt gap, and soft merge."""
    if duration <= 0 or total_budget <= 0:
        raise ValueError("duration and frame budget must be positive")

    if not candidates or duration <= 45.0:
        return [(GroundingContext(0.0, duration), total_budget)]

    import math
    import os

    def _cand_score(c: CandidateProposal) -> float:
        return float(getattr(c, "peak_z", 0.0) + 0.5 * max(0.0, getattr(c, "penalized_score", 0.0)))

    # NMS candidates by IoU to get distinct action peaks
    distinct_cands: list[CandidateProposal] = []
    for c in sorted(candidates, key=_cand_score, reverse=True):
        if not any(
            (max(0.0, min(c.end, d.end) - max(c.start, d.start)) / (max(c.end, d.end) - min(c.start, d.start) + 1e-6)) > 0.3
            for d in distinct_cands
        ):
            distinct_cands.append(c)

    distinct_cands = sorted(distinct_cands, key=lambda x: x.start)
    if not distinct_cands:
        return [(GroundingContext(0.0, duration), total_budget)]

    # Adaptive Margin: clamp(3.5 * ln(duration), 8.0, 22.0)
    margin = float(np.clip(3.5 * np.log(max(10.0, duration)), 8.0, 22.0))
    # Adaptive Gap: max(15.0, 3.5 * sqrt(duration))
    gap_threshold = max(15.0, 3.5 * math.sqrt(duration))

    cand_span = distinct_cands[-1].end - distinct_cands[0].start
    if cand_span <= 1.5 * gap_threshold or len(distinct_cands) == 1:
        c_start = distinct_cands[0].start
        c_end = distinct_cands[-1].end
        w_start = max(0.0, c_start - margin)
        w_end = min(duration, c_end + margin)
        if w_end - w_start < min_window:
            mid = (w_start + w_end) / 2.0
            w_start = max(0.0, mid - min_window / 2.0)
            w_end = min(duration, w_start + min_window)
            w_start = max(0.0, w_end - min_window)
        return [(GroundingContext(w_start, w_end), total_budget)]

    # Cluster distinct candidates using adaptive gap threshold
    clusters: list[list[CandidateProposal]] = [[distinct_cands[0]]]
    for c in distinct_cands[1:]:
        if c.start - clusters[-1][-1].end <= gap_threshold:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    # Keep up to top clusters (default 2 windows)
    max_clusters = int(os.environ.get("SGDE_MAX_CLUSTERS", "2"))
    if len(clusters) > max_clusters:
        clusters = sorted(clusters, key=lambda cl: max(_cand_score(x) for x in cl), reverse=True)[:max_clusters]
        clusters = sorted(clusters, key=lambda cl: cl[0].start)

    windows: list[tuple[GroundingContext, int]] = []
    budget_per_win = int(os.environ.get("SGDE_BUDGET_PER_WINDOW", "0")) or max(24, total_budget // len(clusters))

    for cl in clusters:
        c_start = cl[0].start
        c_end = cl[-1].end
        w_start = max(0.0, c_start - margin)
        w_end = min(duration, c_end + margin)
        if w_end - w_start < min_window:
            mid = (w_start + w_end) / 2.0
            w_start = max(0.0, mid - min_window / 2.0)
            w_end = min(duration, w_start + min_window)
            w_start = max(0.0, w_end - min_window)
        windows.append((GroundingContext(w_start, w_end), budget_per_win))

    # Soft Continuity Merge
    if len(windows) == 2:
        w1_ctx, _ = windows[0]
        w2_ctx, _ = windows[1]
        gap_between = w2_ctx.start - w1_ctx.end
        span_ratio = (w2_ctx.end - w1_ctx.start) / duration
        if gap_between <= gap_threshold or span_ratio >= 0.70:
            return [(GroundingContext(w1_ctx.start, w2_ctx.end), total_budget)]

    return windows


def plan_adaptive_sgde_windows(
    timeline: Any,
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    base_budget: int = FRAME_BUDGET,
    fallback_budget: int = 128,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    adaptive_budget: bool = True,
) -> tuple[list[tuple[GroundingContext, int]], str]:
    """Decide between zoomed candidate windows or full exploration."""
    if duration <= 0:
        raise ValueError("duration must be positive")

    if candidates and len(candidates) > 0:
        windows = plan_sgde_windows(
            candidates,
            duration,
            total_budget=base_budget,
            context_seconds=context_seconds,
        )
        return windows, "scout-zoom"
    else:
        budget = fallback_budget if adaptive_budget else base_budget
        return [(GroundingContext(0.0, duration), budget)], "full-video-fallback"


def plan_sgde_corridor(
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    budget: int = FRAME_BUDGET,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
) -> tuple[tuple[Observation, ...], GroundingContext]:
    """Compatibility wrapper returning single corridor observations and context."""
    if not candidates:
        timestamps = uniform_timestamps(0.0, duration, budget)
        return tuple(Observation(t, "exploration") for t in timestamps), GroundingContext(0.0, duration)

    windows = plan_sgde_windows(
        candidates,
        duration,
        total_budget=budget,
        context_seconds=context_seconds,
    )
    context, win_budget = windows[0]
    timestamps = uniform_timestamps(context.start, context.end, win_budget)
    c_start = min(c.start for c in candidates)
    c_end = max(c.end for c in candidates)
    observations = []
    for t in timestamps:
        if c_start <= t <= c_end:
            role = "candidate"
        elif t < c_start:
            role = "pre_context"
        else:
            role = "post_context"
        observations.append(Observation(t, role))
    return tuple(observations), context


def plan_adaptive_sgde_corridor(
    timeline: Any,
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    base_budget: int = FRAME_BUDGET,
    fallback_budget: int = 128,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    adaptive_budget: bool = True,
) -> tuple[tuple[Observation, ...], GroundingContext, str]:
    """Compatibility wrapper returning single corridor observations, context, and route mode."""
    windows, mode = plan_adaptive_sgde_windows(
        timeline,
        candidates,
        duration,
        base_budget=base_budget,
        fallback_budget=fallback_budget,
        context_seconds=context_seconds,
        adaptive_budget=adaptive_budget,
    )
    context, win_budget = windows[0]
    timestamps = uniform_timestamps(context.start, context.end, win_budget)
    obs = tuple(Observation(t, "candidate" if mode == "scout-zoom" else "exploration") for t in timestamps)
    return obs, context, mode


def plan_sgde_evidence(
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    budget: int = FRAME_BUDGET,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    num_anchors: int = DEFAULT_NUM_ANCHORS,
) -> tuple[Observation, ...]:
    """Compatibility wrapper returning observations tuple."""
    obs, _ = plan_sgde_corridor(
        candidates,
        duration,
        budget=budget,
        context_seconds=context_seconds,
    )
    return obs


def assign_observation_roles(
    evidence: TemporalEvidence,
    observations: Sequence[Observation],
) -> None:
    """Map encoded evidence rows back to nearest planned observation roles."""
    if not observations:
        evidence.roles = ()
        evidence.metadata["observation_plan"] = []
        return
    times = np.asarray([o.timestamp for o in observations], dtype=np.float32)
    evidence_times = np.asarray(evidence.timestamps, dtype=np.float32)
    nearest_indices = np.argmin(np.abs(times[:, None] - evidence_times[None, :]), axis=0)
    roles = [observations[idx].role for idx in nearest_indices]
    evidence.roles = tuple(roles)
    evidence.metadata["observation_plan"] = [o.to_dict() for o in observations]
