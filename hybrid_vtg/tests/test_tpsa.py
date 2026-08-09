import pytest
import torch

from hybrid_vtg.config import SpatialAllocatorConfig
from hybrid_vtg.tpsa import (
    TimelinePreservingSpatialAllocator,
    directional_boundary_evidence,
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


def test_directional_boundary_quota_is_assigned_exactly_when_bands_exist():
    tokens = torch.zeros(12, 4, 4)
    tokens[:4, :, 0] = -1
    tokens[4:9, :, 0] = 1
    tokens[9:, :, 0] = -1
    for patch in range(4):
        tokens[:, patch, patch % 3 + 1] = 0.1 * (patch + 1)
    result = _allocator("tpsa_boundary", 0.75)(
        tokens, 2, 2, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), fps=1.0,
    )
    remaining = result.target_tokens - tokens.shape[0]
    quota = int(remaining * 0.10)
    assert result.start_indices.numel() == quota
    assert result.end_indices.numel() == quota
    assert result.start_bands[0].center_frame == 4
    assert result.end_bands[0].center_frame == 9


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
    evidence = torch.zeros(30)
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


def test_local_object_translation_produces_motion_evidence():
    height = width = 5
    generator = torch.Generator().manual_seed(23)
    first = torch.randn(height * width, 16, generator=generator)
    second = first.clone()
    source, destination = 2 * width + 1, 2 * width + 2
    second[destination] = first[source]
    second[source] = torch.randn(16, generator=generator)
    novelty = feature_transition_maps(torch.stack((first, second)), height, width, radius=2)
    assert float(novelty[0, source]) > float(novelty[0].median())
    assert float(novelty[1, destination]) > float(novelty[1].median())


def test_global_translation_is_camera_compensated_away_from_edges():
    height = width = 5
    base = torch.eye(height * width)
    shifted = torch.zeros_like(base)
    for row in range(height):
        for column in range(width - 1):
            shifted[row * width + column + 1] = base[row * width + column]
    novelty = feature_transition_maps(torch.stack((base, shifted)), height, width, radius=2)
    interior = novelty.reshape(2, height, width)[:, :, 1:]
    assert float((interior == 0).float().mean()) >= 0.5
