"""Deterministic global-plus-corridor evidence planning for ASGDE OMTG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...media import uniform_timestamps
from ..sgde_64.proposals import CandidateProposal

BASE_BUDGET = 64
MULTI_PEAK_BUDGET = 128
BASE_ANCHORS = 12
MULTI_PEAK_ANCHORS = 16


@dataclass(frozen=True)
class Corridor:
    id: int
    start: float
    end: float
    score: float
    peak_z: float
    uncertainty: float
    sources: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "start": round(self.start, 3), "end": round(self.end, 3),
                "score": round(self.score, 3), "peak_z": round(self.peak_z, 3),
                "uncertainty": round(self.uncertainty, 3), "sources": list(self.sources)}


@dataclass(frozen=True)
class Observation:
    timestamp: float
    role: str
    corridor_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {"timestamp": round(self.timestamp, 3), "role": self.role, "corridor_id": self.corridor_id}


def merge_candidate_corridors(
    candidates: Sequence[CandidateProposal], duration: float, *, adjacent_seconds: float = 3.0
) -> tuple[Corridor, ...]:
    """Merge overlapping/context-adjacent proposals into continuous corridors."""
    if not candidates:
        return ()
    ordered = sorted(candidates, key=lambda value: (value.start, value.end, -value.score, value.source))
    groups: list[list[CandidateProposal]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        group = groups[-1]
        if candidate.start <= max(value.end for value in group) + adjacent_seconds:
            group.append(candidate)
        else:
            groups.append([candidate])
    corridors = []
    for index, group in enumerate(groups):
        start = max(0.0, min(value.start for value in group))
        end = min(duration, max(value.end for value in group))
        best = max(group, key=lambda value: (value.score, value.peak_z, -value.start))
        corridors.append(Corridor(index, start, end, best.score, best.peak_z,
                                 max(0.0, 1.0 - min(best.peak_z / 3.0, 1.0)),
                                 tuple(sorted({value.source for value in group}))))
    return tuple(corridors)


def select_corridors(corridors: Sequence[Corridor], *, max_corridors: int = 4) -> tuple[Corridor, ...]:
    """Keep high scoring separated corridors with deterministic diversity tie breaks."""
    ranked = sorted(corridors, key=lambda value: (-value.score, -value.peak_z, value.start, value.end))
    selected: list[Corridor] = []
    while ranked and len(selected) < max_corridors:
        if not selected:
            selected.append(ranked.pop(0))
            continue
        choice = max(ranked, key=lambda value: (min(abs((value.start + value.end) / 2 - (other.start + other.end) / 2) for other in selected), value.score, value.peak_z, -value.start))
        selected.append(choice)
        ranked.remove(choice)
    return tuple(sorted(selected, key=lambda value: value.start))


def _add(values: dict[float, Observation], observation: Observation) -> None:
    key = round(observation.timestamp, 8)
    existing = values.get(key)
    if existing is None or observation.role == "global_anchor":
        values[key] = observation


def _padded(corridor: Corridor, duration: float) -> tuple[float, float, float]:
    halo = min(12.0, max(2.0, corridor.duration * (0.2 + 0.3 * corridor.uncertainty)))
    return max(0.0, corridor.start - halo), min(duration, corridor.end + halo), halo


def plan_asgde_evidence(
    corridors: Sequence[Corridor], duration: float, *, budget: int, anchors: int
) -> tuple[Observation, ...]:
    """Allocate anchors first, then local evidence, while enforcing an exact hard budget."""
    if duration <= 0 or budget <= 0 or anchors > budget:
        raise ValueError("invalid ASGDE duration or budget")
    if not corridors:
        return tuple(Observation(timestamp, "exploration") for timestamp in uniform_timestamps(0.0, duration, budget))
    values: dict[float, Observation] = {}
    for timestamp in uniform_timestamps(0.0, duration, anchors):
        _add(values, Observation(timestamp, "global_anchor"))
    remaining = budget - len(values)
    weights = [max(0.001, value.score * (1.0 + value.uncertainty)) for value in corridors]
    raw = [remaining * weight / sum(weights) for weight in weights]
    counts = [max(5, int(value)) for value in raw]
    while sum(counts) > remaining:
        index = max(range(len(counts)), key=lambda i: (counts[i], -i))
        if counts[index] <= 5:
            break
        counts[index] -= 1
    for index in sorted(range(len(counts)), key=lambda i: (-(raw[i] - int(raw[i])), i)):
        if sum(counts) >= remaining:
            break
        counts[index] += 1
    for corridor, count in zip(corridors, counts):
        start, end, halo = _padded(corridor, duration)
        roles = ("pre_context", "onset_transition", "candidate_interior", "offset_transition", "post_context")
        for index, timestamp in enumerate(uniform_timestamps(start, end, count)):
            fraction = (timestamp - start) / max(end - start, 1e-6)
            role = roles[0] if timestamp < corridor.start else roles[-1] if timestamp > corridor.end else roles[min(3, max(1, int(fraction * 4)))]
            _add(values, Observation(timestamp, role, corridor.id))
    # Refill only from eligible global/local evidence regions, preserving the hard budget.
    eligible = [(0.0, duration, None)] + [(*_padded(value, duration)[:2], value.id) for value in corridors]
    grid = max(budget * 16, 512)
    for start, end, corridor_id in eligible:
        for timestamp in uniform_timestamps(start, end, grid):
            _add(values, Observation(timestamp, "candidate_interior" if corridor_id is not None else "global_anchor", corridor_id))
            if len(values) >= budget:
                break
        if len(values) >= budget:
            break
    selected = sorted(values.values(), key=lambda value: (value.timestamp, value.corridor_id is None))[:budget]
    if len(selected) != budget:
        raise RuntimeError(f"planned {len(selected)} observations for {budget}-frame ASGDE budget")
    return tuple(selected)
