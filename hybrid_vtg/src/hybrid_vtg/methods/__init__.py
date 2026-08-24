"""Inference method registration."""

from __future__ import annotations

from typing import Any


def register_methods(registry: Any) -> None:
    from .anchored_corridor_64 import AnchoredCorridor64
    from .coarse_to_fine_64 import CoarseToFine64
    from .native import Native
    from .sgde_64 import SGDE64, ScoutProvider

    registry.register(AnchoredCorridor64.name, AnchoredCorridor64)
    registry.register(CoarseToFine64.name, CoarseToFine64)
    registry.register(Native.name, Native)
    import os
    from ..scout_features import DEFAULT_MODEL

    registry.register(
        SGDE64.name,
        lambda feature_roots=(), scout_provider=None, scout_model=None, **kwargs: (
            SGDE64(
                scout_provider=scout_provider
                or ScoutProvider(
                    model_id=scout_model or os.environ.get("TIMELENS_SCOUT_MODEL", DEFAULT_MODEL),
                    feature_roots=feature_roots,
                )
            )
        ),
    )
