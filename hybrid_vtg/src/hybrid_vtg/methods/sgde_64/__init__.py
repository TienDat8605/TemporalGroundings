"""Scout-Guided Dense Evidence Grounding (SGDE-64, Idea 3)."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Sequence

from tqdm import tqdm

from ...contracts import Method, ModelBackend, Prediction, Sample
from .planning import (
    DEFAULT_CONTEXT_SECONDS,
    DEFAULT_NUM_ANCHORS,
    FRAME_BUDGET,
    Observation,
    assign_observation_roles,
    plan_adaptive_sgde_corridor,
    plan_sgde_evidence,
)
from .proposals import CandidateProposal, extract_candidate_proposals
from .scout import ScoutProvider, ScoutTimeline


class SGDE64(Method):
    """Two-stage scout-guided dense evidence grounding with adaptive frame budget."""

    name = "sgde-64"

    def __init__(
        self,
        scout_provider: ScoutProvider | None = None,
        *,
        frame_budget: int = FRAME_BUDGET,
        fallback_budget: int = 128,
        context_seconds: float = DEFAULT_CONTEXT_SECONDS,
        num_anchors: int = DEFAULT_NUM_ANCHORS,
        adaptive_budget: bool = True,
    ) -> None:
        self.scout_provider = scout_provider or ScoutProvider()
        self.frame_budget = frame_budget
        self.fallback_budget = fallback_budget
        self.context_seconds = context_seconds
        self.num_anchors = num_anchors
        self.adaptive_budget = adaptive_budget
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

        # Stage 2: Adaptive Corridor & Frame Planning
        observations, context, route_mode = plan_adaptive_sgde_corridor(
            timeline,
            candidates,
            sample.duration,
            base_budget=self.frame_budget,
            fallback_budget=self.fallback_budget,
            context_seconds=self.context_seconds,
            adaptive_budget=self.adaptive_budget,
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
            "candidates": [c.to_dict() for c in (candidates if route_mode == "scout-zoom" else [])],
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
