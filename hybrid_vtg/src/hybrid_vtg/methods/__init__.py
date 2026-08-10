"""The only three inference methods exposed by the project."""

from __future__ import annotations

from typing import Any


def register_methods(registry: Any) -> None:
    from .coarse_to_fine_64 import CoarseToFine64
    from .hmve import HMVE
    from .tpsa_query import TPSAQuery

    registry.register(CoarseToFine64.name, CoarseToFine64)
    registry.register(TPSAQuery.name, TPSAQuery)
    registry.register(HMVE.name, HMVE)
