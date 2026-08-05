import numpy as np

from hybrid_vtg.config import CoarseConfig
from hybrid_vtg.index import CoarseIndex
from hybrid_vtg.temporal import multiscale_candidates, route, select_candidates, Candidate


def test_route_selects_query_relevant_region_with_halo():
    timestamps = np.arange(0.5, 40.0, 1.0)
    features = np.zeros((len(timestamps), 2), dtype=np.float32)
    features[:, 1] = 1.0
    features[(timestamps >= 16) & (timestamps < 24)] = [1.0, 0.0]
    index = CoarseIndex("video.mp4", "fingerprint", "encoder", 1.0, 40.0, timestamps, features)
    config = CoarseConfig(
        scales=(8.0,), stride_ratio=0.5, union_budget_seconds=12.0,
        maximum_candidates=1, halo_seconds=2.0,
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
        union_budget_seconds=40.0, maximum_candidates=4, halo_seconds=0.0,
        low_confidence_margin=0.01,
    )
    result = select_candidates(candidates, 60.0, config)
    assert result.low_confidence_fallback
    assert len(result.selected_candidates) == 4


def test_multiscale_candidates_include_video_tail():
    candidates = multiscale_candidates(23.0, (8.0,), 0.5)
    assert any(item.start == 15.0 and item.end == 23.0 for item in candidates)
