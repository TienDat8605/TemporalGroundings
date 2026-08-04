import numpy as np

from vlmeval.boundary_refinement import BoundaryRefinementConfig, refine_interval
from vlmeval.proposal_scorer import evidence_curve, propose_intervals


def test_proposal_scorer_finds_high_evidence_run():
    timestamps = np.arange(0.0, 10.0, 0.5)
    query = np.zeros((len(timestamps), 1, 1), dtype=np.float32)
    query[6:12] = 1.0
    motion = np.zeros_like(query)
    mask = np.ones_like(query, dtype=bool)
    proposals = propose_intervals(
        timestamps,
        evidence_curve(query, motion, mask),
        cardinality='single',
    )
    assert len(proposals) == 1
    assert proposals[0].start <= 3.5
    assert proposals[0].end >= 5.5


def test_refinement_stays_in_radius_and_component():
    timestamps = np.arange(0.0, 10.0, 0.25)
    evidence = (timestamps >= 4.0).astype(np.float32)
    continuity = np.zeros_like(evidence)
    continuity[np.argmin(np.abs(timestamps - 4.0))] = 1.0
    result = refine_interval(
        [3.5, 8.0],
        timestamps,
        evidence,
        continuity,
        duration=10.0,
        component=[2.0, 9.0],
        config=BoundaryRefinementConfig(radius_seconds=1.0, minimum_gain=0.0),
    )
    assert 2.5 <= result.refined[0] <= 4.5
    assert 7.0 <= result.refined[1] <= 9.0
    assert result.refined[0] < result.refined[1]
