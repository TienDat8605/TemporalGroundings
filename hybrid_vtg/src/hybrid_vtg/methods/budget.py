"""Shared sampled-frame budget helpers for duration-matched methods."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..contracts import ModelBackend, TemporalEvidence

CELL_SECONDS = 8.0


def duration_budget(duration: float) -> int:
    """Return the locked two-frames-per-eight-second-cell budget."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    value = max(64, 2 * math.ceil(duration / CELL_SECONDS))
    return value + value % 2


def requires_even_frames(model: ModelBackend) -> bool:
    """Qwen-style spatial evidence is produced from temporal tubelets."""
    return "spatial-evidence" in model.capabilities


def pad_even(timestamps: Sequence[float], required: bool) -> tuple[tuple[float, ...], int]:
    values = tuple(float(value) for value in timestamps)
    if required and len(values) % 2:
        return values + (values[-1],), 1
    return values, 0


def scout_timestamps(duration: float, require_even: bool = False) -> tuple[tuple[float, ...], int]:
    """Return one center timestamp for each fixed eight-second cell."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    cells = math.ceil(duration / CELL_SECONDS)
    timestamps = tuple(
        (start + min(start + CELL_SECONDS, duration)) / 2.0
        for start in (index * CELL_SECONDS for index in range(cells))
    )
    return pad_even(timestamps, require_even)


def duplicate_tubelets(
    timestamps: Sequence[float],
    roles: Sequence[str],
    qwen_tubelets: bool,
) -> tuple[tuple[float, ...], tuple[str, ...], int]:
    """Duplicate logical observations into two-frame identical Qwen tubelets."""
    values = tuple(float(value) for value in timestamps)
    values_roles = tuple(roles)
    if len(values) != len(values_roles):
        raise ValueError("timestamps and roles must align")
    if not qwen_tubelets:
        return values, values_roles, 0
    duplicated_timestamps = tuple(value for value in values for _ in range(2))
    duplicated_roles = tuple(role for role in values_roles for _ in range(2))
    return duplicated_timestamps, duplicated_roles, len(values)


@dataclass
class BudgetLedger:
    budget: int
    requested_frames: int = 0
    duplicate_padding_frames: int = 0
    tubelet_duplicate_frames: int = 0

    @property
    def remaining_frames(self) -> int:
        return self.budget - self.requested_frames

    def reserve(
        self,
        timestamps: Sequence[float],
        duplicate_padding: int = 0,
        tubelet_duplicates: int = 0,
    ) -> None:
        requested = len(timestamps)
        if duplicate_padding < 0 or tubelet_duplicates < 0:
            raise ValueError("duplicate frame counts cannot be negative")
        if duplicate_padding + tubelet_duplicates > requested:
            raise ValueError("duplicate frame counts exceed requested frames")
        if requested > self.remaining_frames:
            raise AssertionError(
                f"sampled-frame route requested {self.requested_frames + requested} frames "
                f"for budget {self.budget}"
            )
        self.requested_frames += requested
        self.duplicate_padding_frames += duplicate_padding
        self.tubelet_duplicate_frames += tubelet_duplicates

    def to_dict(self) -> dict[str, int]:
        return {
            "budget": self.budget,
            "requested_frames": self.requested_frames,
            "duplicate_padding_frames": self.duplicate_padding_frames,
            "tubelet_duplicate_frames": self.tubelet_duplicate_frames,
            "remaining_frames": self.remaining_frames,
        }


def temporal_anchor_indices(evidence: TemporalEvidence, scores, maximum: int) -> set[int]:
    """Protect deterministic score-ranked anchors spread over the full timeline."""
    if maximum <= 0:
        return set()
    band_count = min(maximum, evidence.size)
    low, high = min(evidence.timestamps), max(evidence.timestamps)
    if high <= low:
        return {int(scores.float().argmax().item())}
    width = (high - low + 1e-6) / band_count
    anchors: set[int] = set()
    for band in range(band_count):
        start = low + band * width
        end = low + (band + 1) * width
        members = [
            index
            for index, timestamp in enumerate(evidence.timestamps)
            if start <= timestamp < end or (band == band_count - 1 and timestamp <= end)
        ]
        if members:
            anchors.add(max(members, key=lambda index: (float(scores[index]), -index)))
    return anchors
