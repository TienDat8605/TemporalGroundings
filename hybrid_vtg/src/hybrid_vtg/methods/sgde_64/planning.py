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
    """Cluster candidates into 1 to 2 disjoint contiguous windows and allocate frames."""
    if duration <= 0 or total_budget <= 0:
        raise ValueError("duration and frame budget must be positive")

    if not candidates or duration <= 45.0:
        return [(GroundingContext(0.0, duration), total_budget)]

    # NMS candidates by IoU to get distinct action peaks
    distinct_cands: list[CandidateProposal] = []
    for c in sorted(candidates, key=lambda x: getattr(x, "peak_z", 0.0), reverse=True):
        if not any(
            (max(0.0, min(c.end, d.end) - max(c.start, d.start)) / (max(c.end, d.end) - min(c.start, d.start) + 1e-6)) > 0.3
            for d in distinct_cands
        ):
            distinct_cands.append(c)

    distinct_cands = sorted(distinct_cands, key=lambda x: x.start)
    if not distinct_cands:
        return [(GroundingContext(0.0, duration), total_budget)]

    cand_span = distinct_cands[-1].end - distinct_cands[0].start
    if cand_span <= 60.0 or len(distinct_cands) == 1:
        c_start = distinct_cands[0].start
        c_end = distinct_cands[-1].end
        margin = max(context_seconds, min(12.0, (c_end - c_start) * 0.25))
        w_start = max(0.0, c_start - margin)
        w_end = min(duration, c_end + margin)
        if w_end - w_start < min_window:
            mid = (w_start + w_end) / 2.0
            w_start = max(0.0, mid - min_window / 2.0)
            w_end = min(duration, w_start + min_window)
            w_start = max(0.0, w_end - min_window)
        return [(GroundingContext(w_start, w_end), total_budget)]

    # Cluster distinct candidates that are close (<= 20s gap)
    clusters: list[list[CandidateProposal]] = [[distinct_cands[0]]]
    for c in distinct_cands[1:]:
        if c.start - clusters[-1][-1].end <= 20.0:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    # Keep up to top 2 clusters
    if len(clusters) > 2:
        clusters = sorted(clusters, key=lambda cl: max(getattr(x, "peak_z", 0.0) for x in cl), reverse=True)[:2]
        clusters = sorted(clusters, key=lambda cl: cl[0].start)

    windows: list[tuple[GroundingContext, int]] = []
    budget_per_win = max(24, total_budget // len(clusters))

    for cl in clusters:
        c_start = cl[0].start
        c_end = cl[-1].end
        margin = max(context_seconds, min(10.0, (c_end - c_start) * 0.25))
        w_start = max(0.0, c_start - margin)
        w_end = min(duration, c_end + margin)
        if w_end - w_start < min_window:
            mid = (w_start + w_end) / 2.0
            w_start = max(0.0, mid - min_window / 2.0)
            w_end = min(duration, w_start + min_window)
            w_start = max(0.0, w_end - min_window)
        windows.append((GroundingContext(w_start, w_end), budget_per_win))

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
    """Decide between zoomed candidate windows (base_budget) or full exploration (fallback_budget)."""
    if duration <= 0:
        raise ValueError("duration must be positive")

    peak_z = getattr(timeline, "peak_z", 0.0) if timeline is not None else 0.0
    cand_dur = (candidates[0].end - candidates[0].start) if candidates else 0.0

    is_sharp = bool(
        candidates
        and len(candidates) > 0
        and peak_z >= 1.6
        and cand_dur <= 45.0
    )

    if is_sharp:
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
