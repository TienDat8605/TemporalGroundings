"""Evidence planning and frame budget allocation for SGDE (Idea 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...contracts import TemporalEvidence
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


def plan_sgde_evidence(
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    budget: int = FRAME_BUDGET,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    num_anchors: int = DEFAULT_NUM_ANCHORS,
) -> tuple[Observation, ...]:
    """Allocate exactly 64 frames across the scout-guided candidate corridor window."""
    if duration <= 0 or budget <= 0:
        raise ValueError("duration and frame budget must be positive")

    # If no candidates provided, fail-open to uniform full-video exploration
    if not candidates:
        timestamps = uniform_timestamps(0.0, duration, budget)
        return tuple(Observation(t, "exploration") for t in timestamps)

    c_start = min(c.start for c in candidates)
    c_end = max(c.end for c in candidates)
    w_start = max(0.0, c_start - context_seconds)
    w_end = min(duration, c_end + context_seconds)
    min_window = min(20.0, duration)
    if w_end - w_start < min_window:
        mid = (w_start + w_end) / 2.0
        w_start = max(0.0, mid - min_window / 2.0)
        w_end = min(duration, w_start + min_window)
        w_start = max(0.0, w_end - min_window)

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

    return tuple(observations)


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
