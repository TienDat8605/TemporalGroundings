import pytest
import torch

from hybrid_vtg.config import SpatialAllocatorConfig
from hybrid_vtg.tpsa import (
    TimelinePreservingSpatialAllocator,
    directional_boundary_evidence,
    effective_temporal_fps,
    feature_transition_maps,
    select_boundary_bands,
)


def _tokens(frames=8, height=3, width=4, dimension=12):
    generator = torch.Generator().manual_seed(17)
    return torch.randn(frames, height * width, dimension, generator=generator)


def _allocator(policy="tpsa_boundary", ratio=0.25):
    return TimelinePreservingSpatialAllocator(
        SpatialAllocatorConfig(spatial_policy=policy, retention_ratio=ratio),
    )


@pytest.mark.parametrize("policy", ("tpsa_query", "tpsa_motion", "tpsa_boundary"))
@pytest.mark.parametrize("ratio", (0.0, 0.125, 0.25, 1.0))
def test_exact_budget_coverage_order_coordinates_and_no_duplicates(policy, ratio):
    tokens = _tokens()
    query = tokens[3, :2].clone()
    result = _allocator(policy, ratio)(tokens, 3, 4, query, fps=2.0)
    expected = min(tokens.shape[0] * tokens.shape[1], max(round(ratio * tokens.shape[0] * tokens.shape[1]), tokens.shape[0]))
    assert result.target_tokens == expected
    assert result.actual_tokens == expected
    assert result.keep_indices.unique().numel() == expected
    assert torch.equal(result.keep_indices, result.keep_indices.sort().values)
    assert int(result.keep_indices.min()) >= 0
    assert int(result.keep_indices.max()) < tokens.shape[0] * tokens.shape[1]
    assert torch.all(result.frame_allocation >= 1)
    roles = torch.cat((
        result.prototype_indices, result.start_indices, result.end_indices, result.adaptive_indices,
    ))
    assert roles.unique().numel() == expected


def test_flat_evidence_transfers_boundary_quota_and_allocates_uniformly():
    tokens = torch.ones(6, 9, 8)
    result = _allocator("tpsa_boundary", 0.5)(tokens, 3, 3, torch.ones(2, 8), fps=2.0)
    assert result.start_indices.numel() == 0
    assert result.end_indices.numel() == 0
    assert not result.start_bands and not result.end_bands
    assert result.adaptive_indices.numel() == result.target_tokens - tokens.shape[0]
    assert int(result.frame_allocation.max() - result.frame_allocation.min()) <= 1


def test_strong_directional_transitions_retain_both_sides():
    tokens = torch.zeros(12, 20, 4)
    tokens[:4, :, 0] = -1
    tokens[4:9, :, 0] = 1
    tokens[9:, :, 0] = -1
    for patch in range(20):
        tokens[:, patch, patch % 3 + 1] = 0.1 * (patch + 1)
    result = _allocator("tpsa_boundary", 0.75)(
        tokens, 4, 5, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), fps=1.0,
    )
    remaining = result.target_tokens - tokens.shape[0]
    boundary_quota = int(int(remaining * 0.10) * 0.50)
    assert result.start_indices.numel() + result.end_indices.numel() == boundary_quota
    assert result.start_indices.numel() >= 2
    assert result.end_indices.numel() >= 2
    assert result.start_bands[0].center_frame == 4
    assert result.end_bands[0].center_frame == 9
    retained_frames = result.keep_indices // tokens.shape[1]
    start_frames = retained_frames[(retained_frames >= 3) & (retained_frames <= 5)]
    end_frames = retained_frames[(retained_frames >= 8) & (retained_frames <= 10)]
    assert bool((start_frames < 4).any()) and bool((start_frames >= 4).any())
    assert bool((end_frames < 9).any()) and bool((end_frames >= 9).any())


def test_single_frame_uses_zero_motion_and_is_deterministic():
    tokens = _tokens(frames=1, height=2, width=3)
    allocator = _allocator("tpsa_boundary", 0.5)
    first = allocator(tokens, 2, 3, tokens[0, :2], fps=2.0)
    second = allocator(tokens, 2, 3, tokens[0, :2], fps=2.0)
    assert torch.count_nonzero(first.novelty) == 0
    assert not first.start_bands and not first.end_bands
    assert torch.equal(first.keep_indices, second.keep_indices)
    assert first.stats()["per_frame_allocation"] == second.stats()["per_frame_allocation"]


def test_full_retention_is_the_dense_chronological_sequence():
    tokens = _tokens(frames=3, height=2, width=3)
    result = _allocator("tpsa_boundary", 1.0)(tokens, 2, 3, tokens[0, :2], fps=2.0)
    assert torch.equal(result.keep_indices, torch.arange(tokens.shape[0] * tokens.shape[1]))


def test_grid_mismatch_fails_without_budget_rewrite():
    with pytest.raises(ValueError, match="H\\*W=.*but P=12"):
        _allocator("tpsa_query")(_tokens(), 2, 5, _tokens()[0, :2], fps=2.0)


def test_directional_start_rise_end_fall_and_one_second_expansion():
    relevance = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    novelty = torch.zeros_like(relevance)
    novelty[5] = novelty[11] = 1
    start, end = directional_boundary_evidence(relevance, novelty, fps=1.0)
    assert int(start.argmax()) == 5 and float(start[5]) > 0
    assert int(end.argmax()) == 11 and float(end[11]) > 0
    start_bands = select_boundary_bands(start, "start", 1.0, 4.0, 1.0, 4)
    end_bands = select_boundary_bands(end, "end", 1.0, 4.0, 1.0, 4)
    assert (start_bands[0].start_frame, start_bands[0].end_frame) == (4, 6)
    assert (end_bands[0].start_frame, end_bands[0].end_frame) == (10, 12)


def test_four_second_nms_and_four_band_cap():
    evidence = torch.full((30,), 0.1)
    evidence[[2, 4, 8, 12, 16, 20, 24]] = torch.tensor([0.8, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7])
    bands = select_boundary_bands(evidence, "start", 1.0, 4.0, 1.0, 4)
    assert len(bands) == 4
    centers = [band.center_frame for band in bands]
    assert all(right - left >= 4 for left, right in zip(centers, centers[1:]))


def test_static_video_has_zero_feature_native_motion():
    frame = _tokens(frames=1, height=3, width=3)[0]
    video = frame.unsqueeze(0).repeat(4, 1, 1)
    novelty = feature_transition_maps(video, 3, 3)
    assert torch.count_nonzero(novelty) == 0


def test_local_state_change_exceeds_static_background():
    frame = _tokens(frames=1, height=3, width=3)[0]
    video = frame.unsqueeze(0).repeat(3, 1, 1)
    video[1, 4] = -video[1, 4]
    novelty = feature_transition_maps(video, 3, 3)
    assert float(novelty[:, 4].max()) > float(novelty[:, 0].min())


def test_motion_uses_adjacent_same_cell_feature_difference():
    first = torch.zeros(4, 3)
    second = first.clone()
    second[2, 1] = 3.0
    novelty = feature_transition_maps(torch.stack((first, second)), 2, 2)
    assert torch.equal(novelty[:, 2], torch.tensor([3.0, 3.0]))
    assert torch.count_nonzero(novelty[:, [0, 1, 3]]) == 0


def test_motion_magnitude_is_comparable_across_the_complete_video():
    frame = _tokens(frames=1, height=3, width=3)[0]
    video = frame.unsqueeze(0).repeat(4, 1, 1)
    video[2, 4] = -video[2, 4]
    novelty = feature_transition_maps(video, 3, 3)
    frame_strength = novelty.topk(2, dim=1).values.mean(dim=1)
    assert float(frame_strength[1]) > float(frame_strength[0])
    assert float(frame_strength[2]) == float(frame_strength[3])


def test_auxiliary_policies_keep_at_least_ninety_percent_query_overlap():
    tokens = _tokens(frames=10, height=3, width=4)
    query = tokens[3, :2].clone()
    for policy in ("tpsa_motion", "tpsa_boundary"):
        result = _allocator(policy, 0.25)(tokens, 3, 4, query, fps=1.0)
        remaining = result.target_tokens - tokens.shape[0]
        overlap = result.stats()["query_only_nonprototype_overlap_tokens"]
        assert overlap >= remaining - int(remaining * 0.10)
        assert result.stats()["query_only_nonprototype_overlap_fraction"] >= 0.90


def test_query_policy_indices_are_bit_exact_v2_control():
    tokens = _tokens()
    query = tokens[3, :2].clone()
    result = _allocator("tpsa_query", 0.25)(tokens, 3, 4, query, fps=2.0)
    assert result.keep_indices.tolist() == [
        1, 2, 10, 17, 18, 19, 20, 23, 26, 28, 36, 37,
        38, 43, 44, 47, 52, 62, 67, 69, 71, 75, 77, 86,
    ]


def test_motion_preserves_exact_query_frame_allocation():
    tokens = _tokens(frames=10)
    query = tokens[3, :2].clone()
    baseline = _allocator("tpsa_query", 0.5)(tokens, 3, 4, query, fps=1.0)
    motion = _allocator("tpsa_motion", 0.5)(tokens, 3, 4, query, fps=1.0)
    assert torch.equal(motion.frame_allocation, baseline.frame_allocation)


@pytest.mark.parametrize("scale", (0.0, 1e-7))
def test_static_or_weak_motion_falls_back_exactly_to_query(scale):
    frame = _tokens(frames=1, height=3, width=4)[0]
    offsets = torch.arange(6, dtype=frame.dtype)[:, None, None] * scale
    tokens = frame.unsqueeze(0).repeat(6, 1, 1) + offsets
    query = frame[:2].clone()
    baseline = _allocator("tpsa_query", 0.5)(tokens, 3, 4, query, fps=1.0)
    motion = _allocator("tpsa_motion", 0.5)(tokens, 3, 4, query, fps=1.0)
    assert torch.equal(motion.keep_indices, baseline.keep_indices)
    assert motion.stats()["actual_motion_replacements"] == 0
    assert motion.stats()["motion_gated_frame_count"] == tokens.shape[0]


def test_query_negative_background_change_cannot_enter_motion_selection():
    tokens = torch.zeros(6, 8, 3)
    tokens[:, :, 0] = 1.0
    tokens[:, :, 1] = torch.arange(8).float() / 20
    tokens[:, 7, 0] = -100.0
    tokens[3:, 7, 2] = 100.0
    query = torch.tensor([[1.0, 0.0, 0.0]])
    baseline = _allocator("tpsa_query", 0.5)(tokens, 2, 4, query, fps=1.0)
    motion = _allocator("tpsa_motion", 0.5)(tokens, 2, 4, query, fps=1.0)
    changed = torch.arange(6) * 8 + 7
    assert not bool(torch.isin(changed, baseline.keep_indices).any())
    assert not bool(torch.isin(changed, motion.keep_indices).any())


def test_boundary_noise_receives_no_quota():
    evidence = torch.full((20,), 0.01)
    assert select_boundary_bands(evidence, "start", 1.0, 4.0, 1.0, 4) == ()
    frame = _tokens(frames=1, height=3, width=4)[0]
    tokens = frame.unsqueeze(0).repeat(8, 1, 1)
    result = _allocator("tpsa_boundary", 0.5)(tokens, 3, 4, frame[:2], fps=1.0)
    assert result.actual_boundary_replacements == 0
    assert result.quota_returned_to_query == result.auxiliary_quota


def test_effective_temporal_fps_accounts_for_qwen_tubelets():
    assert effective_temporal_fps(2.0, 2) == 1.0
    with pytest.raises(ValueError):
        effective_temporal_fps(2.0, 0)
