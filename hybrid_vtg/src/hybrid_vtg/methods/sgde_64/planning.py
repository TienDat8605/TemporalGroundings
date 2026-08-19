"""Evidence planning and frame budget allocation for SGDE (Idea 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...contracts import TemporalEvidence
from ...media import uniform_timestamps
from .proposals import CandidateProposal

FRAME_BUDGET = 64
DEFAULT_NUM_ANCHORS = 6
DEFAULT_CONTEXT_SECONDS = 4.0


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


def _add_obs(output: dict[float, Observation], obs: Observation) -> None:
    key = round(obs.timestamp, 6)
    existing = output.get(key)
    if existing is None:
        output[key] = obs
    elif obs.role == "global_anchor" and existing.role != "global_anchor":
        output[key] = obs
    elif obs.role == "boundary_transition" and existing.role not in {"global_anchor", "boundary_transition"}:
        output[key] = obs


def plan_sgde_evidence(
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    budget: int = FRAME_BUDGET,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    num_anchors: int = DEFAULT_NUM_ANCHORS,
) -> tuple[Observation, ...]:
    """Allocate exactly 64 frames across global anchors, context padding, candidate interiors, and boundaries."""
    if duration <= 0 or budget <= 0:
        raise ValueError("duration and frame budget must be positive")

    # If no candidates provided, fail-open to uniform exploration
    if not candidates:
        timestamps = uniform_timestamps(0.0, duration, budget)
        return tuple(Observation(t, "exploration") for t in timestamps)

    observations: dict[float, Observation] = {}

    # 1. Global anchors across full video
    anchor_count = min(num_anchors, budget // 4)
    for t in uniform_timestamps(0.0, duration, anchor_count):
        _add_obs(observations, Observation(t, "global_anchor"))

    # 2. Candidate allocations
    remaining = budget - len(observations)
    if remaining > 0 and candidates:
        per_cand_budget = remaining // len(candidates)
        for cand_idx, cand in enumerate(candidates):
            c_start = max(0.0, cand.start)
            c_end = min(duration, cand.end)
            pre_start = max(0.0, c_start - context_seconds)
            post_end = min(duration, c_end + context_seconds)

            # Boundary transitions (high priority)
            _add_obs(observations, Observation(c_start, "boundary_transition", cand_idx))
            _add_obs(observations, Observation(c_end, "boundary_transition", cand_idx))
            if c_start > 0.5:
                _add_obs(observations, Observation(max(0.0, c_start - 0.5), "boundary_transition", cand_idx))
            if c_end < duration - 0.5:
                _add_obs(observations, Observation(min(duration, c_end + 0.5), "boundary_transition", cand_idx))

            # Pre-context frames
            if c_start > pre_start + 0.2:
                for t in uniform_timestamps(pre_start, c_start, 2):
                    _add_obs(observations, Observation(t, "pre_context", cand_idx))

            # Interior candidate frames
            interior_budget = max(4, per_cand_budget - 6)
            for t in uniform_timestamps(c_start, c_end, interior_budget):
                _add_obs(observations, Observation(t, "candidate", cand_idx))

            # Post-context frames
            if post_end > c_end + 0.2:
                for t in uniform_timestamps(c_end, post_end, 2):
                    _add_obs(observations, Observation(t, "post_context", cand_idx))

    # If exceeding budget, trim lowest-priority observations
    if len(observations) > budget:
        # Priority order: global_anchor > boundary_transition > candidate > pre_context > post_context > exploration
        priority = {
            "global_anchor": 5,
            "boundary_transition": 4,
            "candidate": 3,
            "pre_context": 2,
            "post_context": 2,
            "exploration": 1,
        }
        sorted_obs = sorted(
            observations.values(),
            key=lambda o: (-priority.get(o.role, 0), o.timestamp),
        )
        kept = sorted_obs[:budget]
        observations = {round(o.timestamp, 6): o for o in kept}

    # If below budget, fill deterministically from a fine grid
    if len(observations) < budget:
        for t in uniform_timestamps(0.0, duration, budget * 8):
            _add_obs(observations, Observation(t, "exploration"))
            if len(observations) == budget:
                break

    if len(observations) != budget:
        # Final safety check: if grid rounding missed exact count, pad or slice
        times = sorted(observations.keys())
        if len(times) > budget:
            times = times[:budget]
            observations = {t: observations[t] for t in times}
        elif len(times) < budget:
            for t in uniform_timestamps(0.0, duration, budget):
                if round(t, 6) not in observations:
                    observations[round(t, 6)] = Observation(t, "exploration")
                    if len(observations) == budget:
                        break

    return tuple(sorted(observations.values(), key=lambda o: o.timestamp))


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
