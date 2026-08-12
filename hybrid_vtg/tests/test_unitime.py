from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid_vtg.contracts import GroundingContext, ModelBackend, Prediction, Sample, ScoredSpan, TemporalEvidence
from hybrid_vtg.methods.native import Native
from hybrid_vtg.models.pruning import mage_cell_plan
from hybrid_vtg.models.qwen import QwenEvidenceBackend, _generation_token_budget
from hybrid_vtg.models.timelens import (
    TimeLens7EvidenceBackend,
    _install_mage_qwen25_vision_pruning,
    require_native_video_reader,
)
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


def test_qwen3_uses_shared_4096_cell_budget(tmp_path):
    backend = QwenEvidenceBackend("base", tmp_path, name="qwen3-vl-4b")
    assert "timestamp-interleaved" in backend.capabilities
    assert backend.maximum_evidence_units == 4_096


def test_timelens7_supports_native_and_adaptive_paths_with_4096_budget(tmp_path):
    backend = TimeLens7EvidenceBackend("TencentARC/TimeLens-7B", tmp_path)
    assert backend.maximum_evidence_units == 4_096
    assert "native-video-grounding" in backend.capabilities
    assert "timestamp-interleaved" in backend.capabilities
    assert "unitime-coarse" not in backend.capabilities

    single = Sample("single", "v", Path(__file__), 10.0, "slice the cucumber")
    multi = Sample("multi", "v", Path(__file__), 10.0, "stir", cardinality="multi")
    assert "The event happens in <start time> - <end time> seconds" in backend._instruction(single)
    assert "every separate temporal window" in backend._instruction(multi)


def test_native_timelens_requires_decord_before_model_loading(monkeypatch):
    monkeypatch.setattr("hybrid_vtg.models.timelens.importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"qwen-vl-utils\[decord\]"):
        require_native_video_reader()


def test_qwen25_mage_wrapper_preserves_selected_cell_order():
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
            self.fullatt_block_indexes = []

        def rot_pos_emb(self, grid_thw):
            return torch.zeros((int(grid_thw.prod()), 1))

        def get_window_index(self, grid_thw):
            del grid_thw
            return torch.tensor([1, 0, 3, 2]), [0, 8, 16]

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
    _install_mage_qwen25_vision_pruning(model, 0)
    model.model._mage_prune_plan = mage_cell_plan(
        np.array([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=np.float32),
        merge_size=2,
        retention_ratio=0.5,
    )
    values = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    output = model.model.visual(values, torch.tensor([[2, 2, 4]]))

    assert output.shape == (2, 2)
    assert model.model.visual.blocks[0].seen == (8, [0, 4, 8])
    assert torch.equal(output, torch.stack((values[4:8].mean(0), values[8:12].mean(0))))


def test_multi_occurrence_prompts_keep_nearby_events_separate():
    sample = Sample(
        "multi",
        "omtg",
        Path(__file__),
        100.0,
        "event",
        group="omtg",
        cardinality="multi",
    )
    qwen_prompt = QwenEvidenceBackend._prompt(sample, GroundingContext(0.0, 100.0))
    unitime_prompt = UniTimeEvidenceBackend._instruction(sample)
    coarse_prompt = UniTimeEvidenceBackend._instruction(sample, coarse=True)

    assert "EVERY occurrence" in qwen_prompt
    assert "do not merge" in qwen_prompt
    assert "EVERY separate temporal window" in unitime_prompt
    assert "strict JSON" in unitime_prompt
    assert "do not merge" in unitime_prompt
    assert "EVERY coarse timestamp" in coarse_prompt
    assert "JSON array" in coarse_prompt


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


def test_unitime_sentence_style_output_is_parsed():
    spans = parse_spans("From 15.0 seconds to 18.5 seconds.")
    assert [(span.start, span.end) for span in spans] == [(15.0, 18.5)]


class _NativeUniTimeBackend(ModelBackend):
    name = "unitime"
    capabilities = frozenset({"encoded-evidence", "unitime-coarse"})

    def __init__(self):
        self.encoder_calls = 0
        self.predict_calls = 0

    def encode(self, sample, timestamps):
        del sample
        self.encoder_calls += 1
        return TemporalEvidence(torch.ones((len(timestamps), 2)), tuple(timestamps), len(timestamps))

    def query_scores(self, evidence, query):
        del query
        return torch.ones(evidence.size)

    def coarse_corridor(self, sample, evidence, *, segment_seconds):
        del sample, evidence, segment_seconds
        return GroundingContext(32.0, 64.0), {"coarse_raw_output": "32 seconds"}

    def predict(self, sample, evidence, context):
        del sample, evidence, context
        self.predict_calls += 1
        return Prediction((ScoredSpan(33.0, 40.0),))


class _NativeTimeLensBackend(_NativeUniTimeBackend):
    name = "timelens2-4b"
    capabilities = frozenset({"native-video-grounding"})

    def predict_video(self, sample):
        del sample
        return Prediction((ScoredSpan(3.0, 4.0),), telemetry={"native_whole_video_control": True})


def test_native_dispatches_unitime_to_fixed_coarse_fine_hierarchy():
    backend = _NativeUniTimeBackend()
    result = Native().run(
        Sample("long", "video", Path(__file__), 100.0, "event"),
        backend,
        Path("unused"),
    )
    assert result.telemetry["native_family"] == "unitime"
    assert result.telemetry["corridor"] == {"start": 32.0, "end": 64.0}
    assert backend.encoder_calls == result.telemetry["encoder_calls"] == 2
    assert backend.predict_calls == 1


def test_native_dispatches_timelens_to_whole_video_path():
    result = Native().run(
        Sample("short", "video", Path(__file__), 10.0, "event"),
        _NativeTimeLensBackend(),
        Path("unused"),
    )
    assert result.telemetry["native_whole_video_control"] is True


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
