"""Versioned deterministic benchmark subset selection."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .contracts import Sample

SAMPLER_SCHEMA = "sorted-id-python-random-v1"
SUPPORTED_PERCENTAGES = (10, 20, 100)


def ordered_samples(samples: Sequence[Sample], seed: int) -> list[Sample]:
    ordered = sorted(samples, key=lambda sample: sample.id)
    random.Random(seed).shuffle(ordered)
    return ordered


def subset_samples(samples: Sequence[Sample], percentage: int, seed: int) -> list[Sample]:
    if percentage not in SUPPORTED_PERCENTAGES:
        raise ValueError(f"subset must be one of {SUPPORTED_PERCENTAGES}")
    ordered = ordered_samples(samples, seed)
    count = math.ceil(len(ordered) * percentage / 100)
    return ordered[:count]
