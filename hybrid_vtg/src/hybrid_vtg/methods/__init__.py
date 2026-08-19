"""Inference method registration."""

from __future__ import annotations

from typing import Any


def register_methods(registry: Any) -> None:
    from .anchored_corridor_64 import AnchoredCorridor64
    from .coarse_to_fine_64 import CoarseToFine64
    from .native import Native
    from .sgde_64 import SGDE64, ScoutProvider
    from .asgde_omtg import ASGDEOMTG, SCOUT_MODEL, SCOUT_REVISION

    registry.register(AnchoredCorridor64.name, AnchoredCorridor64)
    registry.register(CoarseToFine64.name, CoarseToFine64)
    registry.register(Native.name, Native)
    registry.register(
        SGDE64.name,
        lambda feature_roots=(), scout_provider=None, **kwargs: (
            SGDE64(scout_provider=scout_provider or ScoutProvider(feature_roots=feature_roots))
        ),
    )
    registry.register(
        ASGDEOMTG.name,
        lambda feature_roots=(), scout_provider=None, **kwargs: ASGDEOMTG(
            scout_provider=scout_provider or ScoutProvider(
                model_id=SCOUT_MODEL, revision=SCOUT_REVISION, fps=1.0, feature_roots=feature_roots
            )
        ),
    )
