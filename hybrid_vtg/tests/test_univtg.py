from pathlib import Path

import numpy as np
import torch

from hybrid_vtg.contracts import Sample
from hybrid_vtg.models.univtg import UniVTGBackend, _checkpoint_path
from hybrid_vtg.models.univtg.features import UniVTGFeatures
from hybrid_vtg.models.univtg.vendor.network import NetworkSpec, UniVTGNetwork


def test_checkpoint_shape_driven_loader_supports_clip_pretraining_family(tmp_path: Path):
    network = UniVTGNetwork(
        NetworkSpec(
            video_dim=514,
            text_dim=512,
            hidden_dim=16,
            encoder_layers=1,
            input_projections=2,
            maximum_video_length=75,
        )
    )
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"model": network.state_dict(), "epoch": 1}, checkpoint)
    backend = UniVTGBackend(str(checkpoint), tmp_path / "cache", "clip-b32")
    assert backend.feature_stack == "clip-b32"
    assert backend.maximum_evidence_units == 75
    assert not backend._network.training


def test_downloaded_univtg_checkpoint_directory_resolves_single_file(tmp_path: Path):
    checkpoint = tmp_path / "download" / "nested" / "model_best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    assert _checkpoint_path(str(tmp_path / "download")) == checkpoint


def test_univtg_network_accepts_sparse_absolute_positions():
    network = UniVTGNetwork(
        NetworkSpec(video_dim=6, text_dim=4, hidden_dim=16, encoder_layers=1, input_projections=1)
    ).eval()
    output = network(
        torch.randn(1, 3, 4),
        torch.ones(1, 3),
        torch.randn(1, 4, 6),
        torch.ones(1, 4),
        video_positions=torch.tensor([[0.05, 0.20, 0.75, 0.95]]),
    )
    assert output["pred_spans"].shape == (1, 4, 2)


def test_official_feature_directories_are_concatenated_and_timestamp_sampled(tmp_path: Path):
    first, second = tmp_path / "slowfast", tmp_path / "clip"
    first.mkdir()
    second.mkdir()
    np.savez(first / "video.npz", features=np.arange(12, dtype=np.float32).reshape(4, 3))
    np.savez(second / "video.npz", features=np.arange(8, dtype=np.float32).reshape(4, 2))
    video = tmp_path / "video.mp4"
    video.touch()
    sample = Sample("q", "video.mp4", video, 8.0, "event")
    provider = UniVTGFeatures(tmp_path / "cache", "slowfast-clip-b32", (first, second))

    features = provider.video(sample, (0.0, 4.0, 8.0))

    assert features.shape == (3, 5)
    np.testing.assert_array_equal(features[:, :3], np.asarray([[0, 1, 2], [6, 7, 8], [9, 10, 11]]))
