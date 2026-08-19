"""Focused unit coverage for absolute-source-time ASGDE OMTG."""

from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid_vtg.contracts import ModelBackend, Prediction, Sample, ScoredSpan, TemporalEvidence
from hybrid_vtg.methods.asgde_omtg import ASGDEOMTG, extract_asgde_candidates
from hybrid_vtg.methods.asgde_omtg.planning import (BASE_ANCHORS, BASE_BUDGET, MULTI_PEAK_ANCHORS,
                                                     MULTI_PEAK_BUDGET, merge_candidate_corridors,
                                                     plan_asgde_evidence, select_corridors)
from hybrid_vtg.methods.sgde_64 import ScoutProvider, ScoutTimeline


def _timeline(peaks: tuple[tuple[int, int], ...]) -> ScoutTimeline:
    timestamps = np.arange(0.5, 120.0, 1.0, dtype=np.float32)
    scores = np.zeros(len(timestamps), dtype=np.float32)
    for start, end in peaks:
        scores[start:end] = np.linspace(0.9, 2.5, end - start)
    return ScoutTimeline(timestamps, scores, scores, scores, 0.0, 0.2, float(scores.max()), "test", True)


def test_asgde_candidate_pool_merges_and_limits_separated_peaks():
    timeline = _timeline(((10, 18), (45, 53), (80, 88), (105, 113)))
    raw, retained = extract_asgde_candidates(timeline, 120.0)
    assert any(value.source == "penalized_interval" for value in raw)
    merged = merge_candidate_corridors(retained, 120.0)
    selected = select_corridors(merged)
    assert 1 <= len(selected) <= 4
    assert tuple(value.start for value in selected) == tuple(sorted(value.start for value in selected))


def test_asgde_plans_exact_budget_and_preserves_corridors():
    timeline = _timeline(((20, 30), (70, 80)))
    _, retained = extract_asgde_candidates(timeline, 120.0)
    corridors = select_corridors(merge_candidate_corridors(retained, 120.0))
    observations = plan_asgde_evidence(corridors, 120.0, budget=MULTI_PEAK_BUDGET, anchors=MULTI_PEAK_ANCHORS)
    assert len(observations) == MULTI_PEAK_BUDGET
    assert len({value.timestamp for value in observations}) == MULTI_PEAK_BUDGET
    assert sum(value.role == "global_anchor" for value in observations) == MULTI_PEAK_ANCHORS
    assert {value.corridor_id for value in observations if value.corridor_id is not None} == {value.id for value in corridors}

    fallback = plan_asgde_evidence((), 120.0, budget=BASE_BUDGET, anchors=BASE_ANCHORS)
    assert len(fallback) == BASE_BUDGET
    assert all(value.role == "exploration" for value in fallback)


class _Scout(ScoutProvider):
    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline

    def compute_timeline(self, sample, cache_root, *, smoothing_window=3):
        return self.timeline


class _Model(ModelBackend):
    name = "mock"

    def __init__(self):
        self.encoder_pruning = "none"
        self.post_pruning = "none"
        self.calls = []

    def encode(self, sample, timestamps):
        self.calls.append(("encode", tuple(timestamps)))
        return TemporalEvidence(torch.ones((len(timestamps), 2)), tuple(timestamps), len(timestamps))

    def query_scores(self, evidence, query):
        return torch.ones(evidence.size)

    def predict(self, sample, evidence, context):
        self.calls.append(("predict", context))
        return Prediction((ScoredSpan(20.0, 30.0), ScoredSpan(70.0, 80.0)), "[[20, 30], [70, 80]]")


def test_asgde_one_call_absolute_context_and_validation(tmp_path):
    method = ASGDEOMTG(scout_provider=_Scout(_timeline(((20, 30), (70, 80)))))
    model = _Model()
    sample = Sample("v::0", "v", Path(__file__), 120.0, "a repeated action", group="omtg", cardinality="multi")
    result = method.run(sample, model, tmp_path)
    assert len(model.calls) == 2
    assert model.calls[1][1].start == 0.0 and model.calls[1][1].end == 120.0
    assert model.calls[1][1].prompt_mode == "asgde-sparse-global"
    assert result.spans == (ScoredSpan(20.0, 30.0), ScoredSpan(70.0, 80.0))
    assert result.telemetry["total_frames"] == MULTI_PEAK_BUDGET
    with pytest.raises(ValueError, match="OMTG"):
        method.run(Sample("x", "v", Path(__file__), 10.0, "x"), model, tmp_path)
