"""Official whole-video inference control for released TimeLens checkpoints."""

from __future__ import annotations

from pathlib import Path

from ...contracts import Method, ModelBackend, Prediction, Sample


class TimeLensNative(Method):
    """Run a TimeLens checkpoint with its published two-FPS video prompt."""

    name = "timelens-native"
    required_capabilities = frozenset({"native-video-grounding"})

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        self.validate_model(model)
        if getattr(model, "encoder_pruning", "none") != "none" or getattr(model, "post_pruning", "none") != "none":
            raise ValueError(
                "timelens-native is a dense native control; use unitime-adaptive to evaluate Mage or SemVID"
            )
        return model.predict_video(sample)  # type: ignore[attr-defined]


__all__ = ["TimeLensNative"]
