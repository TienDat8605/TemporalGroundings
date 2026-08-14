"""Uniform full-timeline baseline at the BGS sampled-frame budget."""

from __future__ import annotations

import math
from pathlib import Path

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample
from ...media import uniform_timestamps
from ..budget import BudgetLedger, duration_budget, temporal_anchor_indices
from ..hmve import pack_evidence

RETENTION_RATIO = 0.25


class UniformBudget(Method):
    name = "uniform-budget"

    def __init__(self, retention_ratio: float = RETENTION_RATIO) -> None:
        if not 0 < retention_ratio <= 1:
            raise ValueError("retention ratio must be in (0, 1]")
        self.retention_ratio = retention_ratio

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        self.validate_model(model)
        ledger = BudgetLedger(duration_budget(sample.duration))
        timestamps = uniform_timestamps(0.0, sample.duration, ledger.budget)
        ledger.reserve(timestamps)

        evidence = model.encode(sample, timestamps)
        evidence.roles = ("global_anchor",) * evidence.size
        scores = model.query_scores(evidence, sample.query)
        target = max(1, round(evidence.size * self.retention_ratio))
        if model.maximum_evidence_units is not None:
            target = min(target, model.maximum_evidence_units)
        anchor_count = min(target, max(2, math.ceil(math.sqrt(target))))
        anchors = temporal_anchor_indices(evidence, scores, anchor_count)
        compact = pack_evidence(evidence, scores, target, anchors)
        prediction = model.predict(sample, compact, GroundingContext(0.0, sample.duration))
        return Prediction(
            prediction.spans,
            prediction.raw_output,
            {
                **prediction.telemetry,
                **ledger.to_dict(),
                "sampled_frames": ledger.requested_frames,
                "encoder_calls": 1,
                "query_scoring_calls": 1,
                "llm_or_fusion_calls": 1,
                "created_evidence": evidence.size,
                "retained_evidence": compact.size,
                "retention_ratio": compact.size / evidence.size,
                "retention_target": self.retention_ratio,
                "protected_anchors": len(anchors),
                "policy": "uniform",
            },
        )
