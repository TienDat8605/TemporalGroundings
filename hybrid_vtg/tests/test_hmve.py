import threading
from types import SimpleNamespace

import pytest
import torch

from hybrid_vtg.config import GrounderConfig, ObservationConfig, SpatialAllocatorConfig
from hybrid_vtg.hmve import (
    EvidenceUnit,
    estimate_vision_transformer_tflops,
    propose_corridors,
    select_evidence,
    unit_times,
)
from hybrid_vtg.semvid_bridge import (
    GroundingRequest,
    PreparedGroundingBatch,
    SemVIDGrounder,
    _HMVEInputUnit,
)
from hybrid_vtg.types import Component, Sample


def _unit(pass_id, time, embeddings, relevance, observation=0):
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return EvidenceUnit(
        pass_id=pass_id,
        absolute_time=time,
        grid_height=1,
        grid_width=embeddings.shape[0],
        source_height=16,
        source_width=16 * embeddings.shape[0],
        embeddings=embeddings,
        query_relevance=torch.tensor(relevance, dtype=torch.float32),
        source_observation=observation,
    )


def test_projected_units_keep_absolute_source_timestamps():
    class Metadata:
        fps = 10.0
        frames_indices = [10, 11, 40, 41]

    assert unit_times(Metadata(), temporal_units=2) == pytest.approx([1.05, 4.05])


def test_vision_compute_estimate_accounts_for_each_observation_grid():
    config = SimpleNamespace(depth=2, hidden_size=4, intermediate_size=8)
    grids = torch.tensor([[1, 2, 2], [1, 1, 2]])
    tokens = 6
    attention_pairs = 4**2 + 2**2
    expected = 2 * (
        8 * tokens * 4**2 + 6 * tokens * 4 * 8 + 4 * attention_pairs * 4
    ) / 1e12

    assert estimate_vision_transformer_tflops(config, grids) == pytest.approx(expected)


def test_query_peaks_produce_multiple_expanded_disjoint_corridors():
    times = torch.arange(0.0, 50.0, 5.0)
    relevance = torch.tensor([0.1, 1.0, 0.2, 0.1, 0.2, 0.3, 0.1, 0.9, 0.2, 0.1])
    corridors = propose_corridors(
        times,
        relevance,
        50.0,
        maximum_corridors=2,
        minimum_seconds=4.0,
        margin_seconds=2.0,
        nms_seconds=10.0,
    )

    assert len(corridors) == 2
    assert corridors[0].end <= corridors[1].start
    assert corridors[0].start <= 5.0 <= corridors[0].end
    assert corridors[1].start <= 35.0 <= corridors[1].end


def test_flat_scout_evidence_falls_back_to_broad_uniform_corridors():
    times = torch.arange(0.0, 100.0, 10.0)
    corridors = propose_corridors(
        times,
        torch.ones_like(times),
        100.0,
        maximum_corridors=3,
        minimum_seconds=4.0,
        margin_seconds=2.0,
        nms_seconds=10.0,
    )

    assert len(corridors) == 3
    assert [value.center for value in corridors] == pytest.approx([0.0, 40.0, 90.0])


def test_accumulator_preserves_every_scout_anchor_and_exact_budget():
    units = [
        _unit(0, 1.0, [[1.0, 0.0], [0.0, 1.0]], [0.1, 0.3]),
        _unit(0, 9.0, [[1.0, 0.0], [0.0, -1.0]], [0.2, 0.4]),
        _unit(1, 1.0, [[1.0, 1.0], [0.0, 1.0]], [0.8, 0.9]),
    ]

    selection = select_evidence(units, target_tokens=4, deduplication_similarity=0.99)

    assert selection.actual_tokens == 4
    assert selection.anchor_tokens == 2
    assert selection.retained_by_pass[0] >= 2
    assert selection.retained_by_pass[1] >= 1
    assert selection.redundant_coarse_tokens == 1
    assert all(indices.tolist() == sorted(indices.tolist()) for indices in selection.local_indices)


def test_accumulator_rejects_a_budget_that_cannot_preserve_global_coverage():
    units = [
        _unit(0, 1.0, [[1.0, 0.0]], [0.1]),
        _unit(0, 2.0, [[0.0, 1.0]], [0.2]),
    ]

    with pytest.raises(ValueError, match="preserve every scout anchor"):
        select_evidence(units, target_tokens=1)


def test_phase_a_uses_two_encoder_calls_and_one_compact_generation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class InnerModel:
        def __init__(self):
            self.encoder_calls = 0
            self.rope_deltas = None

        def get_video_features(self, _pixels, grids, return_token_scores=False):
            assert return_token_scores is True
            self.encoder_calls += 1
            vectors = (
                [
                    torch.tensor([[1.0, 0.0], [0.8, 0.2]]),
                    torch.tensor([[0.0, 1.0], [0.2, 0.8]]),
                ]
                if grids.shape[0] == 2
                else [torch.tensor([[0.7, 0.7], [1.0, 0.0]])]
            )
            return tuple(vectors), None, None

        @staticmethod
        def get_rope_index(input_ids, **_kwargs):
            positions = torch.arange(input_ids.shape[-1]).view(1, 1, -1).expand(3, 1, -1)
            return positions, torch.zeros((1, 1), dtype=torch.long)

    class Model:
        def __init__(self):
            self.model = InnerModel()
            self.config = SimpleNamespace(
                video_token_id=12,
                vision_start_token_id=10,
                vision_end_token_id=11,
                vision_config=SimpleNamespace(depth=2, hidden_size=2, intermediate_size=4),
            )
            self.visual = SimpleNamespace(spatial_merge_size=1)
            self.embedding = torch.nn.Embedding(32, 2)
            self.generations = 0

        def get_input_embeddings(self):
            return self.embedding

        @staticmethod
        def _tflops_video_tokens_batch(counts):
            return float(sum(counts))

        def generate(self, input_ids, inputs_embeds, attention_mask, position_ids, **_kwargs):
            self.generations += 1
            assert inputs_embeds.shape[:2] == input_ids.shape
            assert attention_mask.shape == input_ids.shape
            assert position_ids.shape == (3, 1, input_ids.shape[-1])
            return torch.cat((input_ids, torch.tensor([[20]])), dim=1)

    def unit(pass_id, time, cache_index=-1):
        return _HMVEInputUnit(
            video=torch.zeros((2, 3, 2, 2)),
            metadata=None,
            pass_id=pass_id,
            absolute_time=time,
            source_observation=0,
            cache_index=cache_index,
        )

    grounder = SemVIDGrounder.__new__(SemVIDGrounder)
    grounder.torch = torch
    grounder.model = Model()
    grounder.config = GrounderConfig()
    grounder.spatial_allocator = SpatialAllocatorConfig(retention_ratio=0.5)
    grounder.observation = ObservationConfig(
        policy="hmve", maximum_corridors=1, minimum_corridor_seconds=2.0,
    )
    grounder.generation_config = object()
    grounder._processor_lock = threading.Lock()
    scout_units = (unit(0, 1.0, 0), unit(0, 5.0, 1))
    detailed_units = [unit(1, 3.0)]
    grounder._decode_hmve_corridors = lambda _request, _corridors: detailed_units
    final_ids = torch.tensor([[
        10, 12, 12, 11,
        10, 12, 12, 11,
        10, 12, 12, 11,
        15,
    ]])
    grounder._process_hmve_units = lambda _sample, _units: {
        "input_ids": final_ids,
        "attention_mask": torch.ones_like(final_ids),
        "video_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 2], [1, 1, 2]]),
        "pixel_values_videos": torch.zeros((6, 2)),
    }
    sample = Sample("sample", "video.mp4", "/video.mp4", 10.0, "open drawer")
    request = GroundingRequest(sample, Component(0.0, 10.0, 1.0))
    prepared = PreparedGroundingBatch(
        requests=(request,),
        inputs={},
        prompt_length=0,
        input_stats=({},),
        preparation_seconds=0.0,
        pinned_memory_bytes=0,
        ready_at=0.0,
        hmve_scout_units=scout_units,
        hmve_reference_tokens=8,
    )
    scout_inputs = {
        "input_ids": torch.tensor([[15]]),
        "attention_mask": torch.ones((1, 1), dtype=torch.long),
        "query_ids": torch.tensor([[15]]),
        "query_attention_mask": torch.ones((1, 1), dtype=torch.long),
        "pixel_values_videos": torch.zeros((4, 2)),
        "video_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 2]]),
    }

    _, compact_inputs, compact_length, _, _ = grounder._hmve_generate(
        prepared, scout_inputs, {},
    )

    assert grounder.model.model.encoder_calls == 2
    assert grounder.model.generations == 1
    assert compact_length == 11
    assert compact_inputs["attention_mask"].sum().item() == 11
    stats = grounder.model.fastvid_last_stats
    assert stats["actual_retained_tokens"] == 4
    assert stats["hmve"]["encoder_calls"] == 2
    assert stats["hmve"]["llm_generations"] == 1
    assert stats["hmve"]["cache_reused_scout_units"] == 2
