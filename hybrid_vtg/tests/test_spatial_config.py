import pytest

from hybrid_vtg.cli import _config, _parser
from hybrid_vtg.config import ObservationConfig, SpatialAllocatorConfig


def _args(*extra):
    return _parser().parse_args((
        "run", "--benchmark", "jsonl", "--data", "samples.jsonl", "--output", "out.jsonl", *extra,
    ))


def test_primary_cli_is_full_video_semvid_by_default():
    config = _config(_args())
    assert config.spatial_allocator.spatial_policy == "semvid"
    assert config.observation.policy == "single_pass"
    assert set(config.to_dict()) == {"grounder", "spatial_allocator", "observation"}


def test_staged_tpsa_policy_is_explicit():
    config = _config(_args("--spatial-policy", "tpsa_motion", "--retention-ratio", "0.25"))
    assert config.spatial_allocator.spatial_policy == "tpsa_motion"
    assert config.spatial_allocator.retention_ratio == 0.25


def test_hmve_observation_policy_is_independent_of_spatial_policy():
    config = _config(_args(
        "--observation-policy", "hmve",
        "--spatial-policy", "tpsa_query",
        "--hmve-scout-fps", "0.25",
        "--hmve-scout-pixel-tokens", "1024",
        "--hmve-maximum-corridors", "3",
    ))
    assert config.observation == ObservationConfig(
        policy="hmve",
        scout_fps=0.25,
        scout_total_pixel_tokens=1024,
        maximum_corridors=3,
    )
    assert config.spatial_allocator.spatial_policy == "tpsa_query"


def test_optimized_profile_enables_verified_batch_and_prefetch():
    config = _config(_args("--optimization-profile", "optimized"))
    assert config.grounder.batch_size == 2
    assert config.grounder.preprocess_workers == 1
    assert config.grounder.prefetch_depth == 2


def test_v3_auxiliary_constants_are_explicit_and_conservative():
    config = SpatialAllocatorConfig()
    assert config.auxiliary_fraction == 0.10
    assert config.boundary_share == 0.50
    assert config.motion_query_beta == 0.50
    assert config.evidence_mad_multiplier == 2.0
    with pytest.raises(ValueError, match="auxiliary fraction"):
        SpatialAllocatorConfig(auxiliary_fraction=0.11)
