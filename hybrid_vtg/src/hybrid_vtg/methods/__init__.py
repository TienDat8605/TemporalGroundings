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

    def _make_sgde(frame_budget: int = 64, fallback_budget: int = 128, planning_mode: str = "single_window", name: str = "sgde-64"):
        return lambda feature_roots=(), scout_provider=None, scout_model=None, **kwargs: (
            SGDE64(
                name=name,
                frame_budget=int(os.environ.get("SGDE_FRAME_BUDGET", frame_budget)),
                fallback_budget=int(os.environ.get("SGDE_FALLBACK_BUDGET", fallback_budget)),
                planning_mode=os.environ.get("SGDE_PLANNING_MODE", planning_mode),
                scout_provider=scout_provider
                or ScoutProvider(
                    model_id=scout_model or os.environ.get("TIMELENS_SCOUT_MODEL", DEFAULT_MODEL),
                    feature_roots=feature_roots,
                ),
            )
        )

    registry.register(SGDE64.name, _make_sgde(64, 128, planning_mode="multi_window", name="sgde-64"))
    registry.register("sgde-64-multiwindow", _make_sgde(64, 128, planning_mode="multi_window", name="sgde-64-multiwindow"))
    registry.register("sgde-64-baseline", _make_sgde(64, 128, planning_mode="single_window", name="sgde-64-baseline"))
    registry.register("sgde-128", _make_sgde(128, 256, planning_mode="multi_window", name="sgde-128"))
    registry.register("sgde-128-singlewindow", _make_sgde(128, 256, planning_mode="single_window", name="sgde-128-singlewindow"))
    registry.register("sgde-128-multiwindow", _make_sgde(128, 256, planning_mode="multi_window", name="sgde-128-multiwindow"))
    registry.register("sgde-256", _make_sgde(256, 256, planning_mode="multi_window", name="sgde-256"))
    registry.register("sgde-256-multiwindow", _make_sgde(256, 256, planning_mode="multi_window", name="sgde-256-multiwindow"))
