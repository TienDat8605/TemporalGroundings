"""Inference method registration."""

from __future__ import annotations

from typing import Any


def register_methods(registry: Any) -> None:
    from .anchored_corridor_64 import AnchoredCorridor64
    from .coarse_to_fine_64 import CoarseToFine64
    from .native import Native

    registry.register(AnchoredCorridor64.name, AnchoredCorridor64)
    registry.register(CoarseToFine64.name, CoarseToFine64)
    registry.register(Native.name, Native)
