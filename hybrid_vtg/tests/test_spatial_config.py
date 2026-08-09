import pytest

from hybrid_vtg.cli import _config, _parser
from hybrid_vtg.config import SpatialAllocatorConfig


def _args(*extra):
    return _parser().parse_args((
        "run", "--benchmark", "jsonl", "--data", "samples.jsonl", "--output", "out.jsonl", *extra,
    ))


def test_primary_cli_is_full_video_semvid_by_default():
    config = _config(_args())
    assert config.spatial_allocator.spatial_policy == "semvid"
    assert set(config.to_dict()) == {"grounder", "spatial_allocator"}


def test_staged_tpsa_policy_is_explicit():
    config = _config(_args("--spatial-policy", "tpsa_motion", "--retention-ratio", "0.25"))
    assert config.spatial_allocator.spatial_policy == "tpsa_motion"
    assert config.spatial_allocator.retention_ratio == 0.25


def test_optimized_profile_enables_verified_batch_and_prefetch():
    config = _config(_args("--optimization-profile", "optimized"))
    assert config.grounder.batch_size == 2
    assert config.grounder.preprocess_workers == 1
    assert config.grounder.prefetch_depth == 2


def test_query_core_and_boundary_quotas_cannot_overcommit_budget():
    with pytest.raises(ValueError, match="must not exceed"):
        SpatialAllocatorConfig(query_core_fraction=0.9, boundary_quota_fraction=0.1)
