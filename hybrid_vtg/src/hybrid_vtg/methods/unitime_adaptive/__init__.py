"""Training-free adaptive top-k routing for the frozen UniTime grounder."""

from __future__ import annotations

import math
from pathlib import Path

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample, TemporalEvidence
from ...media import uniform_timestamps
from ..hmve import _temporal_scores, observation_timestamps, pack_evidence, propose_boundary_bands, propose_corridors


class UniTimeAdaptive(Method):
    """Replace UniTime's fixed coarse segment with HMVE-selected corridors.

    The UniTime adapter remains frozen. A cheap full-video scout selects top-k
    query-relevant corridors, which are re-encoded at the adapter's familiar
    two-fps granularity. Boundary bands receive a final high-rate observation,
    then all evidence is passed to one timestamp-interleaved UniTime prefill.
    """

    name = "unitime-adaptive"
    maximum_top_k = 8
    required_capabilities = frozenset({"encoded-evidence", "timestamp-interleaved"})

    def __init__(
        self,
        *,
        top_k: int = 4,
        scout_fps: float = 0.5,
        detail_fps: float = 1.0,
        boundary_fps: float = 2.0,
        boundary_radius: float = 2.0,
        short_seconds: float = 64.0,
    ) -> None:
        if not 1 <= top_k <= self.maximum_top_k:
            raise ValueError(f"top-k corridors must be between 1 and {self.maximum_top_k}")
        if min(scout_fps, detail_fps, boundary_fps, boundary_radius, short_seconds) <= 0:
            raise ValueError("sampling rates, boundary radius, and short threshold must be positive")
        self.top_k = top_k
        self.scout_fps = scout_fps
        self.detail_fps = detail_fps
        self.boundary_fps = boundary_fps
        self.boundary_radius = boundary_radius
        self.short_seconds = short_seconds

    @staticmethod
    def _even_frames(duration: float, fps: float) -> int:
        frames = max(2, math.ceil(duration * fps))
        return frames if frames % 2 == 0 else frames + 1

    @staticmethod
    def _pass(role: str, evidence: TemporalEvidence) -> dict:
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
            frames = self._even_frames(sample.duration, self.detail_fps)
            evidence = model.encode(sample, uniform_timestamps(0.0, sample.duration, frames))
            prediction = model.predict(sample, evidence, GroundingContext(0.0, sample.duration))
            return Prediction(
                prediction.spans,
                prediction.raw_output,
                {
                    **prediction.telemetry,
                    "adaptive_corridors": False,
                    "short_video_bypass": True,
                    "top_k": self.top_k,
                    "scout_frames": 0,
                    "detail_frames": frames,
                    "encoder_calls": 1,
                    "llm_calls": 1,
                    "encoder_passes": [self._pass("fine", evidence)],
                },
            )

        scout_frames = self._even_frames(sample.duration, self.scout_fps)
        # UniTime processes at most 1,024 temporal tubelets in one hierarchy
        # level. Qwen tubelets contain two source frames.
        scout_frames = min(scout_frames, 2_048)
        scout = model.encode(sample, uniform_timestamps(0.0, sample.duration, scout_frames))
        scout.roles = ("global_anchor",) * scout.size
        scout_scores = model.query_scores(scout, sample.query)
        temporal = _temporal_scores(scout, scout_scores)
        corridors = propose_corridors(temporal, sample.duration, maximum=self.top_k)

        detail_timestamps, _ = observation_timestamps(corridors, self.detail_fps)
        detail = model.encode(sample, detail_timestamps)
        detail.roles = ("content",) * detail.size
        detail_scores = model.query_scores(detail, sample.query)
        refined = _temporal_scores(detail, detail_scores)

        bands = propose_boundary_bands(
            refined,
            corridors,
            sample.duration,
            radius=self.boundary_radius,
        )
        boundary_timestamps, _ = observation_timestamps(
            bands,
            self.boundary_fps,
            roles=[band.role for band in bands],
        )
        boundary = model.encode(sample, boundary_timestamps)
        boundary.roles = tuple(
            next(
                (band.role for band in bands if band.start <= timestamp <= band.end),
                "boundary",
            )
            for timestamp in boundary.timestamps
        )

        merged = TemporalEvidence.concatenate([scout, detail, boundary])
        merged_scores = model.query_scores(merged, sample.query)
        anchor_indices: set[int] = set()
        for timestamp, _, _ in temporal:
            candidates = [index for index, value in enumerate(merged.timestamps) if abs(value - timestamp) < 1e-5]
            if candidates:
                anchor_indices.add(max(candidates, key=lambda index: float(merged_scores[index])))

        target = merged.size
        if model.maximum_evidence_units is not None:
            target = min(target, model.maximum_evidence_units)
        evidence = pack_evidence(merged, merged_scores, target, anchor_indices)
        prediction = model.predict(sample, evidence, GroundingContext(0.0, sample.duration))
        return Prediction(
            prediction.spans,
            prediction.raw_output,
            {
                **prediction.telemetry,
                "adaptive_corridors": True,
                "short_video_bypass": False,
                "top_k": self.top_k,
                "corridors": [corridor.__dict__ for corridor in corridors],
                "boundary_bands": [band.__dict__ for band in bands],
                "scout_frames": scout.source_frames,
                "detail_frames": detail.source_frames,
                "boundary_frames": boundary.source_frames,
                "encoder_calls": 3,
                "llm_calls": 1,
                "created_evidence": merged.size,
                "retained_evidence": evidence.size,
                "scout_anchors": len(anchor_indices),
                "absolute_timestamps_preserved": True,
                "encoder_passes": [
                    self._pass("scout", scout),
                    self._pass("corridor", detail),
                    self._pass("boundary", boundary),
                ],
            },
        )


__all__ = ["UniTimeAdaptive"]
