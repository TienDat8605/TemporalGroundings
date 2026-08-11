from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid_vtg.contracts import GroundingContext, ModelBackend, Prediction, Sample, ScoredSpan, TemporalEvidence
from hybrid_vtg.methods.unitime_adaptive import UniTimeAdaptive
from hybrid_vtg.methods.unitime_fixed import UniTimeFixed
from hybrid_vtg.models.pruning import mage_cell_plan
from hybrid_vtg.models.qwen import QwenEvidenceBackend, _generation_token_budget
from hybrid_vtg.models.unitime import (
    UniTimeEvidenceBackend,
    _CompactQwen2Mixin,
    _install_mage_qwen2_vision_pruning,
    adaptive_frame_size,
    compact_mrope_positions,
)
from hybrid_vtg.postprocess import consolidate_spans, parse_spans


def test_compact_mrope_preserves_sparse_row_and_column_coordinates():
    values = torch.tensor([[1, 2, 99, 99, 3, 99, 4]])
    positions = compact_mrope_positions(
        values,
        99,
        [
            [[0, 0, 1], [0, 1, 0]],
            [[0, 2, 3]],
        ],
    )

    assert positions.shape == (3, 1, 7)
    assert positions[:, 0, 2].tolist() == [2, 2, 3]
    assert positions[:, 0, 3].tolist() == [2, 3, 2]
    assert positions[:, 0, 5].tolist() == [5, 7, 8]
    assert positions[:, 0, 6].tolist() == [9, 9, 9]


def test_compact_mrope_preserves_relative_time_inside_coarse_segment():
    values = torch.tensor([[1, 99, 99, 2]])
    positions = compact_mrope_positions(values, 99, [[[5, 0, 0], [6, 0, 0]]])
    assert positions[0, 0, 1:3].tolist() == [1, 2]


def test_qwen2_generation_computes_only_requested_logits():
    class FullLogitsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 7, bias=False)

        def forward(self, inputs_embeds=None, **kwargs):
            del kwargs
            hidden_states = inputs_embeds
            return self.lm_head(hidden_states)

    class CompactModel(_CompactQwen2Mixin, FullLogitsModel):
        pass

    model = CompactModel()
    hidden = torch.ones(1, 16, 4)
    assert model(inputs_embeds=hidden).shape == (1, 16, 7)
    assert model(inputs_embeds=hidden, logits_to_keep=1).shape == (1, 1, 7)
    # The temporary inference hook must not alter later full-forward calls.
    assert model(inputs_embeds=hidden).shape == (1, 16, 7)


def test_adaptive_frame_size_respects_merger_cell_budget_and_aspect():
    width, height = adaptive_frame_size(320, 240, 64)
    assert width % 28 == height % 28 == 0
    assert (width // 28) * (height // 28) <= 64
    assert width > height


def test_qwen2_backends_use_4096_cell_budget_without_requiring_adapter(tmp_path):
    backend = UniTimeEvidenceBackend(None, tmp_path, name="qwen2-vl-7b")
    assert backend.maximum_evidence_units == 4_096
    assert backend.adapter_checkpoint is None


def test_qwen3_adaptive_capability_does_not_change_its_budget(tmp_path):
    backend = QwenEvidenceBackend("base", tmp_path, name="qwen3-vl-4b")
    assert "timestamp-interleaved" in backend.capabilities
    assert backend.maximum_evidence_units is None


def test_qwen2_mage_wrapper_prunes_before_vision_block():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, hidden_states, *, cu_seqlens, position_embeddings, **kwargs):
            self.seen = (hidden_states.shape[0], cu_seqlens.tolist())
            return hidden_states

    class Merger(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states.reshape(-1, 4, hidden_states.shape[-1]).mean(1)

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([Block()])
            self.merger = Merger()

        def rot_pos_emb(self, grid_thw):
            return torch.zeros((int(grid_thw.prod()), 1))

        def forward(self, hidden_states, grid_thw, **kwargs):
            raise AssertionError("dense forward should be replaced")

    class Inner:
        def __init__(self):
            self.visual = Visual()
            self._mage_prune_plan = None

    class Model:
        def __init__(self):
            self.model = Inner()
            vision = type("Vision", (), {"spatial_merge_size": 2})()
            self.config = type("Config", (), {"vision_config": vision})()

    model = Model()
    _install_mage_qwen2_vision_pruning(model, 0)
    model.model._mage_prune_plan = mage_cell_plan(
        np.ones((2, 1, 2), dtype=np.float32),
        merge_size=2,
        retention_ratio=0.5,
    )
    output = model.model.visual(
        torch.arange(32, dtype=torch.float32).reshape(16, 2),
        torch.tensor([[2, 2, 4]]),
    )

    assert output.shape == (2, 2)
    assert model.model.visual.blocks[0].seen == (8, [0, 4, 8])


class _TimestampBackend(ModelBackend):
    name = "timestamp"
    capabilities = frozenset({"encoded-evidence", "timestamp-interleaved"})

    def __init__(self):
        self.encoder_calls = 0
        self.predict_calls = 0
        self.last_evidence = None

    @property
    def maximum_evidence_units(self):
        return 10_000

    def encode(self, sample, timestamps):
        del sample
        self.encoder_calls += 1
        count = len(timestamps)
        embeddings = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)
        return TemporalEvidence(
            embeddings,
            tuple(timestamps),
            count,
            {"cell_coordinates": [[index, 0, 0] for index in range(count)]},
        )

    def query_scores(self, evidence, query):
        del query
        return torch.linspace(0, 1, evidence.size)

    def predict(self, sample, evidence, context: GroundingContext):
        del sample, context
        self.predict_calls += 1
        self.last_evidence = evidence
        return Prediction((ScoredSpan(1.0, 2.0),))


def test_unitime_adaptive_uses_top_k_corridors_and_one_final_generation():
    method = UniTimeAdaptive(top_k=2, short_seconds=64)
    backend = _TimestampBackend()
    sample = Sample("long", "video", Path(__file__), 100.0, "event")

    result = method.run(sample, backend, Path("unused"))

    assert result.telemetry["adaptive_corridors"] is True
    assert result.telemetry["top_k"] == 2
    assert len(result.telemetry["corridors"]) == 2
    assert backend.encoder_calls == 3
    assert backend.predict_calls == 1
    assert backend.last_evidence.timestamps == tuple(sorted(backend.last_evidence.timestamps))
    assert len(backend.last_evidence.metadata["cell_coordinates"]) == backend.last_evidence.size


def test_unitime_adaptive_uses_reduced_sampling_defaults():
    method = UniTimeAdaptive()
    assert (method.scout_fps, method.detail_fps, method.boundary_fps) == (0.5, 1.0, 2.0)


def test_unitime_adaptive_short_video_bypasses_routing():
    method = UniTimeAdaptive(top_k=3, short_seconds=64)
    backend = _TimestampBackend()
    sample = Sample("short", "video", Path(__file__), 20.0, "event")

    result = method.run(sample, backend, Path("unused"))

    assert result.telemetry["short_video_bypass"] is True
    assert backend.encoder_calls == 1
    assert backend.predict_calls == 1


def test_unitime_adaptive_bounds_top_k_to_visual_budget():
    with pytest.raises(ValueError, match="between 1 and 8"):
        UniTimeAdaptive(top_k=9)


class _FixedBackend(_TimestampBackend):
    capabilities = frozenset({"encoded-evidence", "timestamp-interleaved", "unitime-coarse"})

    def coarse_corridor(self, sample, evidence, *, segment_seconds):
        del sample, evidence, segment_seconds
        return GroundingContext(32.0, 64.0), {"coarse_raw_output": "32 seconds"}


def test_unitime_fixed_uses_trained_coarse_then_fine_pass():
    method = UniTimeFixed(short_seconds=64, segment_seconds=32)
    backend = _FixedBackend()
    sample = Sample("long", "video", Path(__file__), 100.0, "event")

    result = method.run(sample, backend, Path("unused"))

    assert result.telemetry["fixed_segment_baseline"] is True
    assert result.telemetry["corridor"] == {"start": 32.0, "end": 64.0}
    assert result.telemetry["encoder_calls"] == 2
    assert result.telemetry["llm_calls"] == 2
    assert backend.encoder_calls == 2
    assert backend.predict_calls == 1


def test_unitime_coarse_prediction_expands_through_last_selected_segment(monkeypatch, tmp_path):
    backend = UniTimeEvidenceBackend("adapter", tmp_path)
    evidence = TemporalEvidence(
        torch.eye(4),
        (0.0, 32.0, 64.0, 96.0),
        8,
        {"cell_coordinates": [[index, 0, 0] for index in range(4)]},
    )
    sample = Sample("long", "video", Path(__file__), 120.0, "event")
    monkeypatch.setattr(
        backend,
        "_evidence_prompt",
        lambda *args, **kwargs: (None, None, None, None, [0.0, 32.0, 64.0, 96.0]),
    )
    monkeypatch.setattr(backend, "_generate", lambda *args: "32 seconds, 64 seconds")

    corridor, telemetry = backend.coarse_corridor(sample, evidence, segment_seconds=32)

    assert corridor == GroundingContext(32.0, 96.0)
    assert telemetry["coarse_selected"] == [32.0, 64.0]
    assert telemetry["coarse_fallback"] is False


def test_unitime_sentence_style_output_is_parsed():
    spans = parse_spans("From 15.0 seconds to 18.5 seconds.")
    assert [(span.start, span.end) for span in spans] == [(15.0, 18.5)]


def test_omtg_gets_larger_generation_budget_only():
    omtg = Sample("1", "v", Path(__file__), 10.0, "event", group="omtg", cardinality="multi")
    tacos = Sample("2", "v", Path(__file__), 10.0, "event", group="tacos")
    assert _generation_token_budget(omtg) == 256
    assert _generation_token_budget(tacos) == 32


def test_close_occurrences_are_not_merged():
    spans = consolidate_spans(
        (ScoredSpan(10.0, 15.0), ScoredSpan(17.0, 22.0)),
        duration=30.0,
    )
    assert [(span.start, span.end) for span in spans] == [(10.0, 15.0), (17.0, 22.0)]
