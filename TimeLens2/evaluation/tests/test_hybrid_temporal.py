import numpy as np

from vlmeval.hybrid_temporal import (
    CoarseIndex,
    multiscale_candidates,
    score_candidates,
    select_candidates,
)
from vlmeval.omtg_search import Window


def test_multiscale_candidates_cover_video_and_include_content_windows():
    candidates = multiscale_candidates(
        73.0,
        scales=(8.0, 16.0, 32.0),
        content_windows=(Window(5.0, 27.0),),
    )
    assert any(candidate.source == 'content' for candidate in candidates)
    for scale in (8.0, 16.0, 32.0):
        selected = [candidate for candidate in candidates if candidate.source == 'multiscale'
                    and candidate.scale == scale]
        assert selected[0].start == 0.0
        assert selected[-1].end == 73.0


def test_scoring_and_selection_are_deterministic_and_budgeted():
    timestamps = np.arange(0.5, 40.0, 1.0)
    features = np.stack([timestamps / 40.0, 1.0 - timestamps / 40.0], axis=1)
    index = CoarseIndex('video.mp4', 'hash', 'checkpoint', 1.0, timestamps, features)
    candidates = multiscale_candidates(40.0, scales=(8.0, 16.0))
    scored = score_candidates(candidates, index, np.asarray([1.0, 0.0]))
    first = select_candidates(
        scored,
        40.0,
        union_budget_seconds=18.0,
        maximum_candidates=3,
        halo_seconds=1.0,
    )
    second = select_candidates(
        scored,
        40.0,
        union_budget_seconds=18.0,
        maximum_candidates=3,
        halo_seconds=1.0,
    )
    assert first == second
    assert first.retained_union_seconds <= 18.0
    assert len(first.selected) <= 3


def test_low_confidence_route_uses_declared_fallback():
    timestamps = np.arange(0.5, 32.0, 1.0)
    features = np.ones((len(timestamps), 4), dtype=np.float32)
    index = CoarseIndex('video.mp4', 'hash', 'checkpoint', 1.0, timestamps, features)
    scored = score_candidates(
        multiscale_candidates(32.0, scales=(8.0,)),
        index,
        np.ones(4, dtype=np.float32),
    )
    route = select_candidates(
        scored,
        32.0,
        union_budget_seconds=24.0,
        maximum_candidates=3,
        low_confidence_margin=0.1,
    )
    assert route.low_confidence_fallback
