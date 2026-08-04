import numpy as np

from vlmeval.spatial_pruning import SpatialPruningConfig, prune_spatial_tokens


def test_spatial_pruning_respects_budget_and_frame_minimum():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(4, 4, 4, 8)).astype(np.float32)
    query = rng.normal(size=8).astype(np.float32)
    result = prune_spatial_tokens(
        features,
        query,
        [0, 0, 1, 1],
        config=SpatialPruningConfig(
            keep_ratio=0.25,
            minimum_tokens_per_frame=2,
            spatial_cells=(1, 1),
        ),
    )
    assert int(result.mask.sum()) == result.budget == 16
    assert np.all(result.mask.sum(axis=(1, 2)) >= 2)
    assert np.isfinite(result.signals.combined).all()


def test_full_retention_keeps_every_token():
    features = np.arange(2 * 2 * 2 * 4, dtype=np.float32).reshape(2, 2, 2, 4)
    result = prune_spatial_tokens(
        features,
        np.ones(4, dtype=np.float32),
        [0, 0],
        config=SpatialPruningConfig(keep_ratio=1.0),
    )
    assert result.mask.all()
    assert len(result.kept_indices) == features.shape[0] * features.shape[1] * features.shape[2]


def test_motion_is_reset_across_component_boundary():
    features = np.zeros((2, 2, 2, 3), dtype=np.float32)
    features[0, ..., 0] = 1.0
    features[1, ..., 1] = 1.0
    result = prune_spatial_tokens(
        features,
        np.ones(3, dtype=np.float32),
        [0, 1],
        config=SpatialPruningConfig(keep_ratio=0.5, spatial_cells=(1, 1)),
    )
    assert np.all(result.signals.motion == 0)
