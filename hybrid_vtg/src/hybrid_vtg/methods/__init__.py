"""Inference method registration."""

from __future__ import annotations

from typing import Any


def register_methods(registry: Any) -> None:
    from .coarse_to_fine_64 import CoarseToFine64
    from .hmve import HMVE
    from .tpsa_query import TPSAQuery
    from .unitime_adaptive import UniTimeAdaptive
    from .unitime_fixed import UniTimeFixed

    registry.register(CoarseToFine64.name, CoarseToFine64)
    registry.register(TPSAQuery.name, TPSAQuery)
    registry.register(HMVE.name, HMVE)
    registry.register(UniTimeAdaptive.name, UniTimeAdaptive)
    registry.register(UniTimeFixed.name, UniTimeFixed)
