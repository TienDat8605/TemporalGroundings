"""Built-in training-free inference methods."""

from __future__ import annotations

from typing import Any


def register_methods(registry: Any) -> None:
    from .boundary_guided_sparsification import BoundaryGuidedSparsification
    from .coarse_to_fine_64 import CoarseToFine64
    from .hmve import HMVE
    from .tpsa_query import TPSAQuery
    from .uniform_budget import UniformBudget

    registry.register(BoundaryGuidedSparsification.name, BoundaryGuidedSparsification)
    registry.register(CoarseToFine64.name, CoarseToFine64)
    registry.register(TPSAQuery.name, TPSAQuery)
    registry.register(HMVE.name, HMVE)
    registry.register(UniformBudget.name, UniformBudget)
