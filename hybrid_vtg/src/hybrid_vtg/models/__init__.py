"""Model backend registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def register_models(registry: Any) -> None:
    from .qwen import QwenEvidenceBackend
    from .univtg import UniVTGBackend

    registry.register(
        "qwen3-vl-4b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), prune_ratio=0.0, prune_layer=12: (
            QwenEvidenceBackend(
                checkpoint or "Qwen/Qwen3-VL-4B-Instruct",
                Path(cache_dir),
                name="qwen3-vl-4b",
                prune_ratio=prune_ratio,
                prune_layer=prune_layer,
            )
        ),
    )
    registry.register(
        "timelens2-4b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), prune_ratio=0.0, prune_layer=12: (
            QwenEvidenceBackend(
                checkpoint or "MCG-NJU/TimeLens2-4B",
                Path(cache_dir),
                name="timelens2-4b",
                prune_ratio=prune_ratio,
                prune_layer=prune_layer,
            )
        ),
    )
    registry.register(
        "univtg",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), prune_ratio=0.0, prune_layer=12: (
            UniVTGBackend(
                checkpoint=checkpoint,
                cache_dir=Path(cache_dir),
                model_spec=model_spec,
                feature_roots=tuple(Path(path) for path in feature_roots),
            )
        ),
    )
