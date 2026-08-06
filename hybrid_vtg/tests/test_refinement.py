import numpy as np

from hybrid_vtg.config import RefinementConfig
from hybrid_vtg.refinement import decide_refinement, refine_from_signals
from hybrid_vtg.types import Component, GroundingPrediction


def _prediction(interval=(3.0, 7.0), component=Component(0.0, 10.0, 1.0)):
    return GroundingPrediction(interval, component, "", {}, {}, {})


def test_adaptive_refinement_uses_fixed_label_free_tiers():
    config = RefinementConfig(adaptive=True)
    high = decide_refinement(
        _prediction(), {"boundary_confidence": 0.95}, config,
        expert_fps=2.0, low_confidence_route=False,
    )
    medium = decide_refinement(
        _prediction(), {"boundary_confidence": 0.80}, config,
        expert_fps=2.0, low_confidence_route=False,
    )
    low = decide_refinement(
        _prediction(), {"boundary_confidence": 0.20}, config,
        expert_fps=2.0, low_confidence_route=False,
    )
    assert not high.refine and high.tier == "high"
    assert medium.refine and medium.fps == 4.0
    assert low.refine and low.fps == 8.0


def test_adaptive_refinement_protects_component_edges_and_uncertain_routes():
    config = RefinementConfig(adaptive=True)
    edge = decide_refinement(
        _prediction((0.25, 7.0)), {"boundary_confidence": 1.0}, config,
        expert_fps=2.0, low_confidence_route=False,
    )
    uncertain = decide_refinement(
        _prediction(), {"boundary_confidence": 1.0}, config,
        expert_fps=2.0, low_confidence_route=True,
    )
    assert edge.refine and edge.tier == "low" and edge.reason == "component_edge"
    assert uncertain.refine and uncertain.tier == "low" and uncertain.reason == "low_confidence_route"


def test_refinement_moves_to_semantic_transitions():
    timestamps = np.arange(0.0, 10.25, 0.25)
    evidence = ((timestamps >= 3.0) & (timestamps < 7.0)).astype(float)
    continuity = np.zeros_like(evidence)
    result = refine_from_signals(
        (2.5, 7.5), timestamps, evidence, continuity,
        duration=10.0, component=Component(0.0, 10.0, 1.0),
        config=RefinementConfig(radius_seconds=1.0, minimum_gain=0.0),
    )
    assert 2.75 <= result.interval[0] <= 3.25
    assert 6.75 <= result.interval[1] <= 7.25


def test_refinement_cannot_leave_component():
    timestamps = np.arange(0.0, 10.5, 0.5)
    evidence = np.linspace(0, 1, len(timestamps))
    result = refine_from_signals(
        (4.0, 6.0), timestamps, evidence, np.ones_like(evidence),
        duration=10.0, component=Component(3.0, 7.0, 1.0),
        config=RefinementConfig(radius_seconds=5.0, minimum_gain=-1.0),
    )
    assert 3.0 <= result.interval[0] < result.interval[1] <= 7.0


def test_unrelated_shot_cut_cannot_attract_boundaries():
    timestamps = np.arange(0.0, 10.25, 0.25)
    evidence = np.ones(len(timestamps))
    continuity = np.zeros(len(timestamps))
    continuity[8] = 10.0
    continuity[32] = 10.0
    result = refine_from_signals(
        (3.0, 7.0), timestamps, evidence, continuity,
        duration=10.0, component=Component(0.0, 10.0, 1.0),
        config=RefinementConfig(radius_seconds=2.0, minimum_gain=0.01),
    )
    assert result.interval == (3.0, 7.0)
    assert not result.start_accepted
    assert not result.end_accepted
