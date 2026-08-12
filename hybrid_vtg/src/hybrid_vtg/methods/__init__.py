"""Inference method registration."""

from __future__ import annotations

from typing import Any


def register_methods(registry: Any) -> None:
    from .coarse_to_fine_64 import CoarseToFine64
    from .native import Native

    registry.register(CoarseToFine64.name, CoarseToFine64)
    registry.register(Native.name, Native)
