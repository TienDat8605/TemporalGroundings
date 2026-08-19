"""Absolute-source-time SGDE for OMTG multi-span grounding."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Sequence

from tqdm import tqdm

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample, TemporalEvidence
from ..sgde_64.proposals import (CandidateProposal, extract_hysteresis_components,
                                 extract_multiscale_density_windows, extract_penalized_intervals, temporal_nms)
from ..sgde_64.scout import ScoutProvider, ScoutTimeline
from .planning import (BASE_ANCHORS, BASE_BUDGET, MULTI_PEAK_ANCHORS, MULTI_PEAK_BUDGET,
                       Corridor, Observation, merge_candidate_corridors, plan_asgde_evidence, select_corridors)

SCOUT_MODEL = "google/siglip2-base-patch16-224"
# Explicit revision keeps ASGDE assets separate from the project-wide default scout.
SCOUT_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def extract_asgde_candidates(timeline: ScoutTimeline, duration: float) -> tuple[list[CandidateProposal], list[CandidateProposal]]:
    if len(timeline) == 0 or duration <= 0 or timeline.mad < 1e-4 or timeline.peak_z < 0.8:
        return [], []
    raw = [*extract_hysteresis_components(timeline.timestamps, timeline.z_scores),
           *extract_penalized_intervals(timeline.timestamps, timeline.z_scores),
           *extract_multiscale_density_windows(timeline.timestamps, timeline.z_scores, duration, scales=(8.0, 16.0, 32.0))]
    retained = temporal_nms(raw, iou_threshold=0.5, max_candidates=max(8, len(raw)))
    return raw, retained


def assign_observation_roles(evidence: TemporalEvidence, observations: Sequence[Observation]) -> None:
    evidence.roles = tuple(next(value.role for value in observations if abs(value.timestamp - timestamp) < 1e-5) for timestamp in evidence.timestamps)
    evidence.metadata["observation_plan"] = [value.to_dict() for value in observations]


class ASGDEOMTG(Method):
    name = "asgde-omtg"

    def __init__(self, scout_provider: ScoutProvider | None = None) -> None:
        self.scout_provider = scout_provider or ScoutProvider(model_id=SCOUT_MODEL, revision=SCOUT_REVISION, fps=1.0)
        self._prepare_root: Path | None = None

    def prepare(self, samples: Sequence[Sample], cache_root: Path) -> None:
        self._prepare_root = cache_root
        import torch
        self.scout_provider.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            for sample in tqdm(samples, desc=f"asgde scout cache ({self.scout_provider.device})", unit="sample"):
                if sample.group != "omtg":
                    raise ValueError("asgde-omtg requires OMTG samples")
                self.scout_provider.compute_timeline(sample, cache_root)
        finally:
            self.scout_provider.unload()

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        self.validate_model(model)
        if sample.group != "omtg" or sample.cardinality != "multi":
            raise ValueError("asgde-omtg requires OMTG multi-span samples")
        if getattr(model, "encoder_pruning", "none") != "none" or getattr(model, "post_pruning", "none") != "none":
            raise ValueError("asgde-omtg requires dense evidence; evaluate Mage and SemVID separately")
        started = perf_counter()
        timeline = self.scout_provider.compute_timeline(sample, self._prepare_root or cache_dir)
        raw, retained = extract_asgde_candidates(timeline, sample.duration)
        merged = merge_candidate_corridors(retained, sample.duration)
        confident = [value for value in merged if value.score >= 0.8 and value.peak_z >= 0.8]
        selected = select_corridors(confident)
        budget = MULTI_PEAK_BUDGET if len(selected) >= 2 else BASE_BUDGET
        anchors = MULTI_PEAK_ANCHORS if budget == MULTI_PEAK_BUDGET else BASE_ANCHORS
        route = "multi-peak" if budget == MULTI_PEAK_BUDGET else "single-peak" if selected else "full-video-fallback"
        observations = plan_asgde_evidence(selected, sample.duration, budget=budget, anchors=anchors)
        evidence = model.encode(sample, tuple(value.timestamp for value in observations))
        assign_observation_roles(evidence, observations)
        result = model.predict(sample, evidence, GroundingContext(0.0, sample.duration, "asgde-sparse-global"))
        roles = {role: sum(value.role == role for value in observations) for role in {value.role for value in observations}}
        return Prediction(result.spans, result.raw_output, {**result.telemetry, "method": self.name, "route_mode": route,
            "route_reason": f"{len(selected)} confident separated corridors", "scout_model": timeline.model_id,
            "scout_cached": timeline.cached, "raw_proposals": [value.to_dict() for value in raw],
            "retained_proposals": [value.to_dict() for value in retained], "merged_corridors": [value.to_dict() for value in merged],
            "selected_corridors": [value.to_dict() for value in selected], "peak_count": len(selected), "total_frames": len(observations),
            "frame_budget": budget, "global_anchors": anchors, "observation_plan": [value.to_dict() for value in observations],
            "observation_role_counts": roles, "encoder_calls": 1, "primary_grounder_calls": 1,
            "timing": {"scout_seconds": perf_counter() - started}})


__all__ = ["ASGDEOMTG", "Corridor", "Observation", "SCOUT_MODEL", "SCOUT_REVISION", "extract_asgde_candidates", "merge_candidate_corridors", "plan_asgde_evidence", "select_corridors"]
