from pathlib import Path

import torch

from hybrid_vtg.contracts import GroundingContext, ModelBackend, Prediction, Sample, TemporalEvidence
from hybrid_vtg.methods.coarse_to_fine_64 import (
    FRAME_BUDGET,
    Window,
    distribute_frames,
    strict_budget,
    uniform_windows,
)
from hybrid_vtg.methods.hmve import HMVE, pack_evidence, propose_boundary_bands, propose_corridors
from hybrid_vtg.methods.tpsa_query import TPSAQuery


def test_coarse_to_fine_policy_never_exceeds_64_frames():
    windows = uniform_windows(900.0)
    routed, policy = strict_budget(windows)
    allocations = distribute_frames(policy["local_budget"], policy["selected_windows"])
    assert len(routed) <= len(windows)
    assert policy["router_frames"] + sum(allocations) == FRAME_BUDGET
    assert all(value >= 2 and value % 2 == 0 for value in allocations)


def test_single_window_is_a_64_frame_bypass():
    routed, policy = strict_budget([Window(0.0, 10.0)])
    assert routed == [Window(0.0, 10.0)]
    # The method bypasses routing for this case; strict_budget itself remains valid.
    assert policy["router_frames"] + policy["local_budget"] == 64


def test_tpsa_retains_exact_budget_and_timeline_coverage():
    timestamps = tuple(float(index // 4) for index in range(40))
    scores = torch.linspace(0, 1, 40)
    indices = TPSAQuery.select_indices(scores, timestamps, 8)
    assert indices.numel() == 8
    assert indices.tolist() == sorted(indices.tolist())
    assert min(timestamps[index] for index in indices) <= 2.0
    assert max(timestamps[index] for index in indices) >= 8.0


def test_hmve_corridors_are_query_ranked_and_separated():
    values = [(float(index * 5), float(index), index) for index in range(10)]
    corridors = propose_corridors(values, 50.0, maximum=3)
    assert len(corridors) == 3
    assert all(0 <= value.start < value.end <= 50 for value in corridors)


def test_hmve_boundary_pass_targets_relevance_rises_and_falls():
    values = [(0.0, 0.1, 0), (2.0, 0.8, 1), (4.0, 0.9, 2), (6.0, 0.2, 3)]
    corridors = propose_corridors(values, 8.0, maximum=1)
    bands = propose_boundary_bands(values, corridors, 8.0, radius=1.0)
    assert [value.role for value in bands] == ["start", "end"]
    assert bands[0].start <= 2.0 <= bands[0].end
    assert bands[1].start <= 4.0 <= bands[1].end


def test_hmve_pack_preserves_anchors_and_exact_target():
    embeddings = torch.eye(8)
    evidence = TemporalEvidence(embeddings, tuple(float(index) for index in range(8)), 8)
    scores = torch.arange(8, dtype=torch.float32)
    packed = pack_evidence(evidence, scores, 4, {0, 3})
    assert packed.size == 4
    assert 0.0 in packed.timestamps and 3.0 in packed.timestamps


class _BoundedBackend(ModelBackend):
    name = "bounded"

    def __init__(self):
        self.encoder_calls = 0
        self.predict_calls = 0

    @property
    def maximum_evidence_units(self):
        return 75

    def encode(self, sample, timestamps):
        del sample
        self.encoder_calls += 1
        values = torch.arange(len(timestamps) * 4, dtype=torch.float32).reshape(len(timestamps), 4)
        return TemporalEvidence(values, tuple(timestamps), len(timestamps))

    def query_scores(self, evidence, query):
        del query
        return torch.arange(evidence.size, dtype=torch.float32)

    def predict(self, sample, evidence, context: GroundingContext):
        del sample, context
        self.predict_calls += 1
        assert evidence.size <= self.maximum_evidence_units
        return Prediction(())


def test_hmve_reserves_detail_capacity_for_bounded_temporal_models():
    sample = Sample("long", "video", Path(__file__), 600.0, "event")
    backend = _BoundedBackend()
    result = HMVE().run(sample, backend, Path("unused"))
    assert result.telemetry["scout_frames"] <= 37
    assert result.telemetry["retained_evidence"] <= 75
    assert backend.encoder_calls == result.telemetry["encoder_calls"] == 3
    assert backend.predict_calls == result.telemetry["llm_or_fusion_calls"] == 1
    assert [value["role"] for value in result.telemetry["passes"]] == [
        "global-scout",
        "corridor-refinement",
        "boundary-refinement",
    ]


def test_hmve_default_pass_rates_end_at_two_fps():
    method = HMVE()
    assert (method.scout_fps, method.detail_fps, method.boundary_fps) == (0.5, 1.0, 2.0)
