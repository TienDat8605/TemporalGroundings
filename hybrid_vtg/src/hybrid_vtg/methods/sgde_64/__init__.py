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
    plan_adaptive_sgde_windows,
    plan_sgde_evidence,
    plan_sgde_windows,
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
        name: str = "sgde-64",
        frame_budget: int = FRAME_BUDGET,
        fallback_budget: int = 128,
        context_seconds: float = DEFAULT_CONTEXT_SECONDS,
        num_anchors: int = DEFAULT_NUM_ANCHORS,
        adaptive_budget: bool = True,
        planning_mode: str = "multi_window",
    ) -> None:
        self.name = name
        self.scout_provider = scout_provider or ScoutProvider()
        self.frame_budget = frame_budget
        self.fallback_budget = fallback_budget
        self.context_seconds = context_seconds
        self.num_anchors = num_anchors
        self.adaptive_budget = adaptive_budget
        self.planning_mode = planning_mode
        self._prepare_root: Path | None = None

    def prepare(self, samples: Sequence[Sample], cache_root: Path) -> None:
        """Batch pre-extraction of scout features on CPU/GPU before main grounder loads."""
        self._prepare_root = cache_root
        import torch

        # Use GPU for batch scout feature extraction if available, then unload
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.scout_provider.device = device
        try:
            self.scout_provider.prepare_batch(samples, cache_root)
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

        # Stage 2: Adaptive Window & Frame Planning
        if self.planning_mode == "single_window":
            obs, ctx, route_mode = plan_adaptive_sgde_corridor(
                timeline,
                candidates,
                sample.duration,
                base_budget=self.frame_budget,
                fallback_budget=self.fallback_budget,
                context_seconds=self.context_seconds,
                adaptive_budget=self.adaptive_budget,
            )
            windows = [(ctx, len(obs))]
        else:
            windows, route_mode = plan_adaptive_sgde_windows(
                timeline,
                candidates,
                sample.duration,
                base_budget=self.frame_budget,
                fallback_budget=self.fallback_budget,
                context_seconds=self.context_seconds,
                adaptive_budget=self.adaptive_budget,
            )

        collected_spans: list[Any] = []
        raw_outputs: list[str] = []
        total_frames = 0
        encode_seconds = 0.0
        predict_seconds = 0.0
        last_result: Any = None
        last_evidence: Any = None

        for context, win_budget in windows:
            from ...media import uniform_timestamps
            timestamps = tuple(uniform_timestamps(context.start, context.end, win_budget))
            total_frames += len(timestamps)
            observations = tuple(Observation(t, "candidate" if route_mode == "scout-zoom" else "exploration") for t in timestamps)

            enc_start = perf_counter()
            evidence = model.encode(sample, timestamps)
            encode_seconds += perf_counter() - enc_start
            last_evidence = evidence

            assign_observation_roles(evidence, observations)
            evidence.metadata["context_start"] = context.start
            evidence.metadata["context_end"] = context.end

            pred_start = perf_counter()
            res = model.predict(sample, evidence, context)
            predict_seconds += perf_counter() - pred_start

            last_result = res
            collected_spans.extend(res.spans)
            if res.raw_output:
                raw_outputs.append(res.raw_output)

        from ...postprocess import consolidate_spans
        if sample.cardinality == "multi":
            final_spans = consolidate_spans(tuple(collected_spans), sample.duration)
        else:
            final_spans = tuple(value for span in collected_spans if (value := span.clipped(sample.duration)))

        dense_tokens = (
            last_evidence.metadata.get("dense_evidence_units", last_evidence.size)
            if last_evidence is not None
            else total_frames
        )
        retained_tokens = (
            last_evidence.metadata.get("encoder_retained_evidence_units", last_evidence.size)
            if last_evidence is not None
            else total_frames
        )
        base_telemetry = last_result.telemetry if last_result is not None else {}

        telemetry = {
            **base_telemetry,
            "method": self.name,
            "route_mode": route_mode,
            "windows_count": len(windows),
            "scout_confident": bool(is_confident),
            "scout_model": str(timeline.model_id),
            "scout_cached": bool(timeline.cached),
            "scout_peak_z": float(round(float(timeline.peak_z), 4)),
            "scout_median": float(round(float(timeline.median), 4)),
            "scout_mad": float(round(float(timeline.mad), 4)),
            "candidates": [c.to_dict() for c in (candidates if route_mode == "scout-zoom" else [])],
            "observation_role_counts": {
                "candidate": int(total_frames if route_mode == "scout-zoom" else 0),
                "exploration": int(total_frames if route_mode != "scout-zoom" else 0),
            },
            "encoder_calls": len(windows),
            "primary_grounder_calls": len(windows),
            "total_frames": int(total_frames),
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
            final_spans,
            "\n---\n".join(raw_outputs) if raw_outputs else "",
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
