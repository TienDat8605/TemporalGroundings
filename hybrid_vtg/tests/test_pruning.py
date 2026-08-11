import numpy as np
import pytest
import torch
from PIL import Image

from hybrid_vtg.contracts import TemporalEvidence
from hybrid_vtg.models.pruning import mage_cell_plan, motion_residual_importance, semvid_select
from hybrid_vtg.models.qwen import QwenEvidenceBackend, _dense_evidence_units, _install_mage_vision_pruning


def test_mage_plan_keeps_exact_complete_cell_budget():
    importance = np.zeros((3, 2, 2), dtype=np.float32)
    importance[1, 1, 1] = 1.0
    importance[2, 0, 1] = 0.8
    plan = mage_cell_plan(importance, merge_size=2, retention_ratio=0.5, anchor_stride=8)

    assert plan.dense_cells == 12
    assert plan.target_cells == 6
    assert sum(plan.cells_per_time) == 6
    assert plan.cells_per_time == (4, 1, 1)
    assert (1, 1, 1) in plan.selected_cells
    assert (2, 0, 1) in plan.selected_cells
    assert len(plan.patch_indices) == 6 * 4
    assert all(
        plan.patch_indices[index : index + 4] == tuple(range(plan.patch_indices[index], plan.patch_indices[index] + 4))
        for index in range(0, len(plan.patch_indices), 4)
    )


def test_motion_residual_importance_tracks_local_change():
    first = np.zeros((64, 64, 3), dtype=np.uint8)
    second = first.copy()
    second[24:40, 32:48] = 255
    values = motion_residual_importance(
        [Image.fromarray(first), Image.fromarray(second)],
        temporal_units=2,
        cell_height=4,
        cell_width=4,
    )

    assert values.shape == (2, 4, 4)
    assert float(values[1].max()) > 0.1
    assert float(values[0].max()) == 0.0


def test_semvid_retains_exact_budget_and_every_time():
    embeddings = torch.eye(4).repeat(3, 1)
    evidence = TemporalEvidence(
        embeddings=embeddings,
        timestamps=(0.0,) * 4 + (1.0,) * 4 + (2.0,) * 4,
        source_frames=6,
        metadata={
            "dense_evidence_units": 12,
            "cell_coordinates": [[time, row, column] for time in range(3) for row in range(2) for column in range(2)],
        },
    )
    compact = semvid_select(
        evidence,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        retention_ratio=0.5,
    )

    assert compact.size == 6
    assert set(compact.timestamps) == {0.0, 1.0, 2.0}
    assert compact.timestamps == tuple(sorted(compact.timestamps))
    assert set(compact.roles) <= {"context", "object", "motion"}
    assert sum(compact.metadata["semvid_role_counts"].values()) == compact.size
    assert len(compact.metadata["cell_coordinates"]) == compact.size


def test_semvid_budget_is_relative_to_dense_encoder_output():
    evidence = TemporalEvidence(
        embeddings=torch.randn(8, 6),
        timestamps=(0.0,) * 3 + (1.0,) * 2 + (2.0,) * 3,
        source_frames=6,
    )
    compact = semvid_select(
        evidence,
        torch.randn(2, 6),
        retention_ratio=0.5,
        dense_evidence_units=12,
    )

    assert compact.size == 6
    assert set(compact.timestamps) == {0.0, 1.0, 2.0}


def test_dense_evidence_count_survives_multi_pass_metadata():
    metadata = {
        "parts": [
            {"dense_evidence_units": 12},
            {"parts": [{"dense_evidence_units": 8}, {"dense_evidence_units": 4}]},
        ]
    }

    assert _dense_evidence_units(metadata, 3) == 24
    assert _dense_evidence_units({}, 3) == 3


def test_mage_wrapper_prunes_before_first_vision_block():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, hidden_states, *, cu_seqlens, position_embeddings, **kwargs):
            self.seen = (hidden_states.shape[0], cu_seqlens.tolist(), hidden_states.dtype)
            return hidden_states

    class Merger(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states.reshape(-1, 4, hidden_states.shape[-1]).mean(dim=1)

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([Block()])
            self.merger = Merger()
            self.deepstack_visual_indexes = []
            self.deepstack_merger_list = torch.nn.ModuleList()

        @property
        def dtype(self):
            return torch.float16

        def fast_pos_embed_interpolate(self, grid_thw):
            return torch.zeros((int(grid_thw.prod()), 2), dtype=self.dtype)

        def rot_pos_emb(self, grid_thw):
            return torch.zeros((int(grid_thw.prod()), 1), dtype=self.dtype)

        def forward(self, hidden_states, grid_thw, **kwargs):
            raise AssertionError("dense path should not be used")

    class Inner:
        def __init__(self):
            self.visual = Visual()
            self._mage_prune_plan = None

        def get_video_features(self, pixel_values_videos, video_grid_thw):
            raise AssertionError("dense path should not be used")

    class Model:
        def __init__(self):
            self.model = Inner()
            self.config = type("Config", (), {"vision_config": type("Vision", (), {"spatial_merge_size": 2})()})()

    model = Model()
    _install_mage_vision_pruning(model, 0)
    model.model._mage_prune_plan = mage_cell_plan(
        np.ones((2, 1, 2), dtype=np.float32),
        merge_size=2,
        retention_ratio=0.5,
    )

    features, deepstack = model.model.get_video_features(
        torch.arange(32, dtype=torch.float32).reshape(16, 2),
        torch.tensor([[2, 2, 4]]),
    )

    assert len(features) == 1
    assert features[0].shape == (2, 2)
    assert deepstack == []
    assert model.model.visual.blocks[0].seen == (8, [0, 4, 8], torch.float16)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"encoder_pruning": "grid"}, "encoder pruning"),
        ({"post_pruning": "topk"}, "post pruning"),
        ({"encoder_retention": 0.0}, "retention ratios"),
        ({"encoder_retention": 0.5}, "encoder retention requires"),
        ({"post_retention": 0.5}, "post retention requires"),
        ({"encoder_pruning": "mage", "encoder_prune_layer": -1}, "non-negative"),
        (
            {
                "encoder_pruning": "mage",
                "encoder_retention": 0.25,
                "post_pruning": "semvid",
                "post_retention": 0.5,
            },
            "cannot exceed",
        ),
    ],
)
def test_qwen_backend_rejects_invalid_pruning_configuration(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        QwenEvidenceBackend("checkpoint", tmp_path, name="qwen", **kwargs)
