from types import SimpleNamespace

import torch

from vlmeval.qwen_sparse_adapter import QwenSparseAdapter, grid_coordinates


def test_grid_coordinates_match_merged_token_count():
    frames, coordinates = grid_coordinates(torch.tensor([[2, 4, 6]]), 2)
    assert len(frames) == 2 * 2 * 3
    assert coordinates.shape == (12, 2)


def test_stock_model_is_rejected_without_sparse_generation_interface():
    model = SimpleNamespace(config=SimpleNamespace(video_token_id=99))
    adapter = QwenSparseAdapter(model)
    audit = adapter.audit(
        torch.ones(4, 8),
        torch.tensor([[1, 99, 99, 99, 99, 2]]),
        torch.tensor([True, False, True, False]),
    )
    assert not audit.compatible
    assert audit.retained_token_count == 2
    assert 'proposal scorer' in audit.reason
