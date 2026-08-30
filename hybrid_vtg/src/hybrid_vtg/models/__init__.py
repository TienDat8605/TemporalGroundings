"""Model backend registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def register_models(registry: Any) -> None:
    from .qwen import QwenEvidenceBackend
    from .timelens import TimeLens7EvidenceBackend
    from .unitime import UniTimeEvidenceBackend
    from .univtg import UniVTGBackend

    registry.register(
        "qwen2-vl-7b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), encoder_pruning="none",
        encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none", post_retention=1.0: (
            UniTimeEvidenceBackend(
                None,
                Path(cache_dir),
                base_checkpoint=checkpoint or "Qwen/Qwen2-VL-7B-Instruct",
                name="qwen2-vl-7b",
                encoder_pruning=encoder_pruning,
                encoder_retention=encoder_retention,
                encoder_prune_layer=encoder_prune_layer,
                post_pruning=post_pruning,
                post_retention=post_retention,
            )
        ),
    )
    registry.register(
        "qwen3-vl-4b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), encoder_pruning="none",
        encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none", post_retention=1.0: (
            QwenEvidenceBackend(
                checkpoint or "Qwen/Qwen3-VL-4B-Instruct",
                Path(cache_dir),
                name="qwen3-vl-4b",
                encoder_pruning=encoder_pruning,
                encoder_retention=encoder_retention,
                encoder_prune_layer=encoder_prune_layer,
                post_pruning=post_pruning,
                post_retention=post_retention,
            )
        ),
    )
    registry.register(
        "qwen3-vl-8b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), encoder_pruning="none",
        encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none", post_retention=1.0: (
            QwenEvidenceBackend(
                checkpoint or "Qwen/Qwen3-VL-8B-Instruct",
                Path(cache_dir),
                name="qwen3-vl-8b",
                encoder_pruning=encoder_pruning,
                encoder_retention=encoder_retention,
                encoder_prune_layer=encoder_prune_layer,
                post_pruning=post_pruning,
                post_retention=post_retention,
                maximum_evidence_units=4_096,
            )
        ),
    )
    registry.register(
        "timelens2-4b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), encoder_pruning="none",
        encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none", post_retention=1.0: (
            QwenEvidenceBackend(
                checkpoint or "MCG-NJU/TimeLens2-4B",
                Path(cache_dir),
                name="timelens2-4b",
                encoder_pruning=encoder_pruning,
                encoder_retention=encoder_retention,
                encoder_prune_layer=encoder_prune_layer,
                post_pruning=post_pruning,
                post_retention=post_retention,
                maximum_evidence_units=4_096,
            )
        ),
    )
    registry.register(
        "timelens-8b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), encoder_pruning="none",
        encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none", post_retention=1.0: (
            QwenEvidenceBackend(
                checkpoint or "TencentARC/TimeLens-8B",
                Path(cache_dir),
                name="timelens-8b",
                encoder_pruning=encoder_pruning,
                encoder_retention=encoder_retention,
                encoder_prune_layer=encoder_prune_layer,
                post_pruning=post_pruning,
                post_retention=post_retention,
                maximum_evidence_units=4_096,
            )
        ),
    )
    registry.register(
        "timelens-7b",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), encoder_pruning="none",
        encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none", post_retention=1.0: (
            TimeLens7EvidenceBackend(
                checkpoint or "TencentARC/TimeLens-7B",
                Path(cache_dir),
                encoder_pruning=encoder_pruning,
                encoder_retention=encoder_retention,
                encoder_prune_layer=encoder_prune_layer,
                post_pruning=post_pruning,
                post_retention=post_retention,
            )
        ),
    )
    registry.register(
        "unitime",
        lambda cache_dir, checkpoint=None, base_checkpoint=None, model_spec=None, feature_roots=(),
        encoder_pruning="none", encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none",
        post_retention=1.0: (
            UniTimeEvidenceBackend(
                checkpoint or "zeqianli/UniTime",
                Path(cache_dir),
                base_checkpoint=base_checkpoint or "Qwen/Qwen2-VL-7B-Instruct",
                encoder_pruning=encoder_pruning,
                encoder_retention=encoder_retention,
                encoder_prune_layer=encoder_prune_layer,
                post_pruning=post_pruning,
                post_retention=post_retention,
            )
        ),
    )
    registry.register(
        "univtg",
        lambda cache_dir, checkpoint=None, model_spec=None, feature_roots=(), encoder_pruning="none",
        encoder_retention=1.0, encoder_prune_layer=0, post_pruning="none", post_retention=1.0: (
            UniVTGBackend(
                checkpoint=checkpoint,
                cache_dir=Path(cache_dir),
                model_spec=model_spec,
                feature_roots=tuple(Path(path) for path in feature_roots),
            )
        ),
    )
