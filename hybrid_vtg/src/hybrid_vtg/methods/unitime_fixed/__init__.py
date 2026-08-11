"""Frozen UniTime's original fixed-segment coarse-to-fine inference."""

from __future__ import annotations

import math
from pathlib import Path

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample
from ...media import uniform_timestamps


class UniTimeFixed(Method):
    """Reference baseline for the public adapter without additional training."""

    name = "unitime-fixed"
    required_capabilities = frozenset({"encoded-evidence", "timestamp-interleaved", "unitime-coarse"})

    def __init__(
        self,
        *,
        fps: float = 2.0,
        short_seconds: float = 64.0,
        segment_seconds: float = 32.0,
    ) -> None:
        if min(fps, short_seconds, segment_seconds) <= 0:
            raise ValueError("FPS and UniTime thresholds must be positive")
        self.fps = fps
        self.short_seconds = short_seconds
        self.segment_seconds = segment_seconds

    def _frames(self, duration: float) -> int:
        frames = min(2_048, max(2, math.ceil(duration * self.fps)))
        return frames if frames % 2 == 0 else frames + 1

    @staticmethod
    def _pass(role: str, evidence) -> dict:
        return {
            "role": role,
            "source_frames": evidence.source_frames,
            "evidence_units": evidence.size,
            "dense_evidence_units": evidence.metadata.get("dense_evidence_units", evidence.size),
            "feature_cache_hit": evidence.metadata.get("feature_cache_hit", False),
            "timing": evidence.metadata.get("timing", {}),
        }

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        self.validate_model(model)
        if sample.duration <= self.short_seconds:
            frames = self._frames(sample.duration)
            evidence = model.encode(sample, uniform_timestamps(0.0, sample.duration, frames))
            prediction = model.predict(sample, evidence, GroundingContext(0.0, sample.duration))
            return Prediction(
                prediction.spans,
                prediction.raw_output,
                {
                    **prediction.telemetry,
                    "fixed_segment_baseline": True,
                    "short_video_bypass": True,
                    "encoder_calls": 1,
                    "llm_calls": 1,
                    "encoder_passes": [self._pass("fine", evidence)],
                },
            )

        coarse_frames = self._frames(sample.duration)
        coarse = model.encode(sample, uniform_timestamps(0.0, sample.duration, coarse_frames))
        corridor, coarse_telemetry = model.coarse_corridor(  # type: ignore[attr-defined]
            sample,
            coarse,
            segment_seconds=self.segment_seconds,
        )
        fine_frames = self._frames(corridor.duration)
        fine = model.encode(sample, uniform_timestamps(corridor.start, corridor.end, fine_frames))
        prediction = model.predict(sample, fine, corridor)
        return Prediction(
            prediction.spans,
            prediction.raw_output,
            {
                **prediction.telemetry,
                **coarse_telemetry,
                "fixed_segment_baseline": True,
                "short_video_bypass": False,
                "segment_seconds": self.segment_seconds,
                "coarse_frames": coarse.source_frames,
                "fine_frames": fine.source_frames,
                "corridor": {"start": corridor.start, "end": corridor.end},
                "encoder_calls": 2,
                "llm_calls": 2,
                "encoder_passes": [self._pass("coarse", coarse), self._pass("fine", fine)],
            },
        )


__all__ = ["UniTimeFixed"]
