"""Evidence planning and frame budget allocation for SGDE (Idea 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def plan_sgde_corridor(
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    budget: int = FRAME_BUDGET,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
) -> tuple[tuple[Observation, ...], GroundingContext]:
    """Compute the corridor GroundingContext and allocate exactly 64 frames uniformly within it."""
    if duration <= 0 or budget <= 0:
        raise ValueError("duration and frame budget must be positive")

    if not candidates or duration <= 45.0:
        timestamps = uniform_timestamps(0.0, duration, budget)
        obs = tuple(Observation(t, "exploration") for t in timestamps)
        return obs, GroundingContext(0.0, duration)

    cand_span = max(c.end for c in candidates) - min(c.start for c in candidates)
    if cand_span <= 80.0:
        c_start = min(c.start for c in candidates)
        c_end = max(c.end for c in candidates)
    else:
        top_cand = candidates[0]
        c_start, c_end = top_cand.start, top_cand.end

    margin = max(context_seconds, min(12.0, (c_end - c_start) * 0.25))
    w_start = max(0.0, c_start - margin)
    w_end = min(duration, c_end + margin)
    min_window = min(30.0, duration)
    if w_end - w_start < min_window:
        mid = (w_start + w_end) / 2.0
        w_start = max(0.0, mid - min_window / 2.0)
        w_end = min(duration, w_start + min_window)
        w_start = max(0.0, w_end - min_window)

    context = GroundingContext(w_start, w_end)
    timestamps = uniform_timestamps(w_start, w_end, budget)
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
    times = np.asarray([o.timestamp for o in observations], dtype=np.float64)
    roles = []
    for t in evidence.timestamps:
        idx = int(np.argmin(np.abs(times - float(t))))
        roles.append(observations[idx].role)
    evidence.roles = tuple(roles)
    evidence.metadata["observation_plan"] = [o.to_dict() for o in observations]
