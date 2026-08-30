"""Checkpoint-native inference for UniTime and TimeLens model families."""

from __future__ import annotations

import math
from pathlib import Path

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample
from ...media import uniform_timestamps


class Native(Method):
    """Use each released model family's native temporal-grounding hierarchy."""

    name = "native"
    required_capabilities = frozenset()

    def __init__(self, *, fps: float = 2.0, short_seconds: float = 64.0, segment_seconds: float = 32.0) -> None:
        if min(fps, short_seconds, segment_seconds) <= 0:
            raise ValueError("FPS and native hierarchy thresholds must be positive")
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

    def _run_unitime(self, sample: Sample, model: ModelBackend) -> Prediction:
        if sample.duration <= self.short_seconds:
            frames = self._frames(sample.duration)
            evidence = model.encode(sample, uniform_timestamps(0.0, sample.duration, frames))
            prediction = model.predict(sample, evidence, GroundingContext(0.0, sample.duration))
            return Prediction(
                prediction.spans,
                prediction.raw_output,
                {
                    **prediction.telemetry,
                    "native_family": "unitime",
                    "fixed_segment_hierarchy": True,
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
                "native_family": "unitime",
                "fixed_segment_hierarchy": True,
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

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        if "unitime-coarse" in model.capabilities:
            return self._run_unitime(sample, model)
        if "native-video-grounding" in model.capabilities:
            if getattr(model, "encoder_pruning", "none") != "none" or getattr(model, "post_pruning", "none") != "none":
                raise ValueError(
                    "native TimeLens inference is dense; use coarse-to-fine-64 for Mage or SemVID"
                )
            return model.predict_video(sample)  # type: ignore[attr-defined]
        raise ValueError(
            f"method 'native' requires a UniTime or TimeLens backend; model {model.name!r} is unsupported"
        )


class NativeFixedFrames(Method):
    """Uniform frame downsampling across the whole unpruned video."""

    def __init__(self, *, frame_budget: int = 128, name: str = "native-128f") -> None:
        self.frame_budget = frame_budget
        self.name = name
        self.required_capabilities = frozenset()

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        if "native-video-grounding" in model.capabilities:
            return model.predict_video(sample, frame_budget=self.frame_budget)  # type: ignore[attr-defined]
        raise ValueError(f"model {model.name!r} does not support native video grounding")


class RandomCropBaseline(Method):
    """Random contiguous temporal corridor matching average candidate span ratio (~40%)."""

    def __init__(self, *, frame_budget: int = 128, crop_ratio: float = 0.40, name: str = "random-crop-128f") -> None:
        self.frame_budget = frame_budget
        self.crop_ratio = crop_ratio
        self.name = name
        self.required_capabilities = frozenset()

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        import random
        rng = random.Random(hash(sample.id) ^ 42)
        crop_dur = max(10.0, sample.duration * self.crop_ratio)
        if crop_dur >= sample.duration:
            start = 0.0
            end = sample.duration
        else:
            start = rng.uniform(0.0, sample.duration - crop_dur)
            end = start + crop_dur
        context = GroundingContext(start, end)
        timestamps = uniform_timestamps(start, end, self.frame_budget)
        evidence = model.encode(sample, timestamps)
        return model.predict(sample, evidence, context)


class OracleCorridorBaseline(Method):
    """Oracle corridor derived from ground-truth target intervals."""

    def __init__(self, *, frame_budget: int = 128, margin_scale: float = 3.5, name: str = "oracle-corridor-128f") -> None:
        self.frame_budget = frame_budget
        self.margin_scale = margin_scale
        self.name = name
        self.required_capabilities = frozenset()

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        import numpy as np
        if not sample.targets:
            start = 0.0
            end = sample.duration
        else:
            margin = float(np.clip(self.margin_scale * np.log(max(10.0, sample.duration)), 8.0, 22.0))
            min_s = min(t[0] for t in sample.targets)
            max_e = max(t[1] for t in sample.targets)
            start = max(0.0, min_s - margin)
            end = min(sample.duration, max_e + margin)
        context = GroundingContext(start, end)
        timestamps = uniform_timestamps(start, end, self.frame_budget)
        evidence = model.encode(sample, timestamps)
        return model.predict(sample, evidence, context)


__all__ = ["Native", "NativeFixedFrames", "RandomCropBaseline", "OracleCorridorBaseline"]
