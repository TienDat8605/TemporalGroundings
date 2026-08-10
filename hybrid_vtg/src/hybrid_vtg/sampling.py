"""Versioned deterministic benchmark subset selection."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .contracts import Sample

SAMPLER_SCHEMA = "sorted-id-python-random-v1"


def validate_percentage(percentage: float) -> float:
    value = float(percentage)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("subset percentage must be between 0 and 100")
    return value


def percentage_key(percentage: float) -> str:
    """Return a sortable, filename-safe key while preserving integer names."""
    value = validate_percentage(percentage)
    whole, separator, fraction = f"{value:.6f}".rstrip("0").rstrip(".").partition(".")
    return f"p{int(whole):03d}" + (f".{fraction}" if separator else "")


def ordered_samples(samples: Sequence[Sample], seed: int) -> list[Sample]:
    ordered = sorted(samples, key=lambda sample: sample.id)
    random.Random(seed).shuffle(ordered)
    return ordered


def subset_samples(samples: Sequence[Sample], percentage: float, seed: int) -> list[Sample]:
    percentage = validate_percentage(percentage)
    ordered = ordered_samples(samples, seed)
    count = math.ceil(len(ordered) * percentage / 100)
    return ordered[:count]
