import numpy as np

from hybrid_vtg.config import CoarseConfig, ProposalConfig
from hybrid_vtg.index import CoarseIndex
from hybrid_vtg.types import Component
from hybrid_vtg.temporal import (
    Candidate,
    interval_boundary_quality,
    multiscale_candidates,
    route,
    select_candidates,
)


def test_route_selects_query_relevant_region_with_halo():
    timestamps = np.arange(0.5, 40.0, 1.0)
    features = np.zeros((len(timestamps), 2), dtype=np.float32)
    features[:, 1] = 1.0
    features[(timestamps >= 16) & (timestamps < 24)] = [1.0, 0.0]
    index = CoarseIndex("video.mp4", "fingerprint", "encoder", 1.0, 40.0, timestamps, features)
    config = CoarseConfig(
        scales=(8.0,), stride_ratio=0.5, union_budget_seconds=12.0,
        maximum_components=1, minimum_halo_seconds=2.0,
        halo_scale_ratio=0.0, maximum_halo_seconds=2.0,
    )
    result, _ = route(index, np.asarray([1.0, 0.0]), config)
    assert result.components[0].start <= 16.0
    assert result.components[0].end >= 24.0
    assert result.retained_union_seconds <= 12.0


def test_low_confidence_route_adds_uniform_coverage():
    candidates = [
        Candidate(index, float(index * 10), float(index * 10 + 8), 8.0, score=0.5)
        for index in range(6)
    ]
    config = CoarseConfig(
        union_budget_seconds=40.0, maximum_components=4, minimum_halo_seconds=0.0,
        halo_scale_ratio=0.0, maximum_halo_seconds=0.0,
        low_confidence_margin=0.01,
    )
    result = select_candidates(candidates, 60.0, config)
    assert result.low_confidence_fallback
    assert len(result.selected_candidates) == 4


def test_multiscale_candidates_include_video_tail():
    candidates = multiscale_candidates(23.0, (8.0,), 0.5)
    assert any(item.start == 15.0 and item.end == 23.0 for item in candidates)


def test_post_halo_duplicates_do_not_consume_component_limit():
    candidates = [
        Candidate(0, 10.0, 18.0, 8.0, score=0.9),
        Candidate(1, 11.0, 19.0, 8.0, score=0.8),
        Candidate(2, 40.0, 48.0, 8.0, score=0.7),
    ]
    config = CoarseConfig(
        union_budget_seconds=30.0, maximum_components=2,
        minimum_halo_seconds=2.0, halo_scale_ratio=0.0, maximum_halo_seconds=2.0,
        low_confidence_margin=0.0,
    )
    result = select_candidates(candidates, 60.0, config)
    assert len(result.components) == 2
    assert 2 in result.selected_candidates


def test_adaptive_halo_is_asymmetric():
    candidate = Candidate(
        0, 10.0, 18.0, 8.0, score=1.0,
        left_uncertainty=1.0, right_uncertainty=0.0,
    )
    config = CoarseConfig(
        fps=2.0, union_budget_seconds=20.0, maximum_components=1,
        minimum_halo_seconds=0.5, halo_scale_ratio=0.0, maximum_halo_seconds=4.0,
        low_confidence_margin=0.0,
    )
    component = select_candidates([candidate], 30.0, config).components[0]
    assert component.start == 9.0
    assert component.end == 18.5


def test_interval_reranking_rewards_clean_boundaries():
    timestamps = np.arange(0.5, 20.0, 1.0)
    features = np.zeros((len(timestamps), 2), dtype=np.float32)
    features[:, 1] = 1.0
    features[(timestamps >= 6) & (timestamps <= 10)] = [1.0, 0.0]
    index = CoarseIndex("video.mp4", "fingerprint", "encoder", 1.0, 20.0, timestamps, features)
    component = Component(0.0, 20.0, 1.0)
    clean = interval_boundary_quality(
        index, np.asarray([1.0, 0.0]), (6.0, 11.0),
        component, ProposalConfig(),
    )
    loose = interval_boundary_quality(
        index, np.asarray([1.0, 0.0]), (2.0, 15.0), component, ProposalConfig(),
    )
    assert clean["score"] > loose["score"]
