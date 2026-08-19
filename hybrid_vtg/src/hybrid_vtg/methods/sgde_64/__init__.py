"""Scout-Guided Dense Evidence Grounding (SGDE-64, Idea 3)."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Sequence

from tqdm import tqdm

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample
from .planning import (
    DEFAULT_CONTEXT_SECONDS,
    DEFAULT_NUM_ANCHORS,
    FRAME_BUDGET,
    Observation,
    assign_observation_roles,
    plan_sgde_evidence,
)
from .proposals import CandidateProposal, extract_candidate_proposals
from .scout import ScoutProvider, ScoutTimeline


class SGDE64(Method):
    """Two-stage scout-guided dense evidence grounding under a strict 64-frame budget."""

    name = "sgde-64"

    def __init__(
        self,
        scout_provider: ScoutProvider | None = None,
        *,
        frame_budget: int = FRAME_BUDGET,
        context_seconds: float = DEFAULT_CONTEXT_SECONDS,
        num_anchors: int = DEFAULT_NUM_ANCHORS,
    ) -> None:
        self.scout_provider = scout_provider or ScoutProvider()
        self.frame_budget = frame_budget
        self.context_seconds = context_seconds
        self.num_anchors = num_anchors
        self._prepare_root: Path | None = None

    def prepare(self, samples: Sequence[Sample], cache_root: Path) -> None:
        """Batch pre-extraction of scout features on CPU/GPU before main grounder loads."""
        self._prepare_root = cache_root
        import torch

        # Use GPU for batch scout feature extraction if available, then unload
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.scout_provider.device = device
        try:
            for sample in tqdm(samples, desc=f"sgde scout cache ({device})", unit="sample"):
                self.scout_provider.compute_timeline(sample, cache_root)
        finally:
            self.scout_provider.unload()

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        self.validate_model(model)
        if getattr(model, "encoder_pruning", "none") != "none" or getattr(
            model, "post_pruning", "none"
        ) != "none":
            raise ValueError(
                "sgde-64 currently requires dense evidence; evaluate Mage and "
                "SemVID as separate follow-up ablations"
            )

        root = self._prepare_root or cache_dir

        # Stage 1: Scout timeline and candidate proposal extraction
        scout_started = perf_counter()
        timeline = self.scout_provider.compute_timeline(sample, root)
        candidates, is_confident = extract_candidate_proposals(
            timeline,
            sample.duration,
            cardinality=sample.cardinality,
        )
        scout_seconds = perf_counter() - scout_started

        if sample.duration <= 10.0:
            route_mode = "single-window-bypass"
            selected_candidates = candidates
        elif is_confident and candidates:
            route_mode = "scout-guided"
            selected_candidates = candidates
        else:
            route_mode = "full-video-fallback"
            selected_candidates = []

        if route_mode == "scout-guided" and selected_candidates:
            if sample.cardinality == "single":
                top_cand = selected_candidates[0]
                c_start, c_end = top_cand.start, top_cand.end
            else:
                c_start = min(c.start for c in selected_candidates)
                c_end = max(c.end for c in selected_candidates)
            margin = max(self.context_seconds, min(12.0, (c_end - c_start) * 0.3))
            w_start = max(0.0, c_start - margin)
            w_end = min(sample.duration, c_end + margin)
            min_window = min(30.0, sample.duration)
            if w_end - w_start < min_window:
                mid = (w_start + w_end) / 2.0
                w_start = max(0.0, mid - min_window / 2.0)
                w_end = min(sample.duration, w_start + min_window)
                w_start = max(0.0, w_end - min_window)
            context = GroundingContext(w_start, w_end)
        else:
            context = GroundingContext(0.0, sample.duration)

        # Stage 2: 64-Frame Evidence Planning
        observations = plan_sgde_evidence(
            selected_candidates,
            sample.duration,
            budget=self.frame_budget,
            context_seconds=self.context_seconds,
            num_anchors=self.num_anchors,
        )
        timestamps = tuple(obs.timestamp for obs in observations)

        # Grounder Encode & Predict
        encode_started = perf_counter()
        evidence = model.encode(sample, timestamps)
        encode_seconds = perf_counter() - encode_started

        assign_observation_roles(evidence, observations)

        predict_started = perf_counter()
        result = model.predict(
            sample,
            evidence,
            context,
        )
        predict_seconds = perf_counter() - predict_started

        role_counts = {
            role: sum(obs.role == role for obs in observations)
            for role in (
                "global_anchor",
                "candidate",
                "pre_context",
                "post_context",
                "boundary_transition",
                "exploration",
            )
        }

        dense_tokens = evidence.metadata.get("dense_evidence_units", evidence.size)
        retained_tokens = evidence.metadata.get("encoder_retained_evidence_units", evidence.size)

        telemetry = {
            **result.telemetry,
            "method": self.name,
            "route_mode": route_mode,
            "scout_confident": bool(is_confident),
            "scout_model": str(timeline.model_id),
            "scout_cached": bool(timeline.cached),
            "scout_peak_z": float(round(float(timeline.peak_z), 4)),
            "scout_median": float(round(float(timeline.median), 4)),
            "scout_mad": float(round(float(timeline.mad), 4)),
            "candidates": [c.to_dict() for c in selected_candidates],
            "observation_role_counts": {k: int(v) for k, v in role_counts.items()},
            "observation_plan": [obs.to_dict() for obs in observations],
            "encoder_calls": 1,
            "primary_grounder_calls": 1,
            "total_frames": int(len(observations)),
            "encoder_dense_tokens": int(dense_tokens),
            "encoder_retained_tokens": int(retained_tokens),
            "timing": {
                "scout_seconds": float(scout_seconds),
                "encode_seconds": float(encode_seconds),
                "predict_seconds": float(predict_seconds),
                "total_seconds": float(scout_seconds + encode_seconds + predict_seconds),
            },
        }

        return Prediction(
            result.spans,
            result.raw_output,
            telemetry,
        )


__all__ = [
    "CandidateProposal",
    "Observation",
    "SGDE64",
    "ScoutProvider",
    "ScoutTimeline",
    "assign_observation_roles",
    "extract_candidate_proposals",
    "plan_sgde_evidence",
]
