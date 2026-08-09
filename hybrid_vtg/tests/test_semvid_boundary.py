from pathlib import Path

import pytest
import torch

from hybrid_vtg.config import GrounderConfig, SpatialAllocatorConfig
from hybrid_vtg.semvid_bridge import (
    SemVIDGrounder, _per_frame_allocation, _render_generation_prompt, validate_semvid_root,
)
from hybrid_vtg.types import Component, Sample


def test_missing_semvid_submodule_has_actionable_error(tmp_path: Path):
    with pytest.raises(RuntimeError, match="git submodule update"):
        validate_semvid_root(tmp_path)


def test_qwen_thinking_is_closed_before_generation():
    class Processor:
        @staticmethod
        def apply_chat_template(prompt, tokenize, add_generation_prompt):
            assert prompt == [{"role": "user"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "rendered"

    assert _render_generation_prompt(Processor(), [{"role": "user"}], True) == "rendered</think>"
    assert _render_generation_prompt(Processor(), [{"role": "user"}], False) == "rendered"


def test_grounding_prompt_is_timestamp_only():
    grounder = SemVIDGrounder.__new__(SemVIDGrounder)
    grounder.config = GrounderConfig()
    sample = Sample("sample", "video.mp4", "/video.mp4", 100.0, "open the drawer")
    prompt = grounder._prompt(sample, Component(20.0, 40.0, 0.5))
    instruction = prompt[0]["content"][1]["text"]
    assert "Localize the event" in instruction
    assert '"start": number, "end": number' in instruction
    assert '"present"' not in instruction


def test_tpsa_prompt_is_timestamp_only_for_one_full_video_component():
    grounder = SemVIDGrounder.__new__(SemVIDGrounder)
    grounder.config = GrounderConfig()
    grounder.spatial_allocator = SpatialAllocatorConfig(spatial_policy="tpsa_boundary")
    sample = Sample("sample", "video.mp4", "/video.mp4", 100.0, "open the drawer")
    instruction = grounder._prompt(sample, Component(0.0, 100.0, 1.0))[0]["content"][1]["text"]
    assert "Localize the event" in instruction
    assert '"start": number, "end": number' in instruction
    assert '"present"' not in instruction


def test_omtg_prompt_requests_all_disjoint_occurrences():
    grounder = SemVIDGrounder.__new__(SemVIDGrounder)
    grounder.config = GrounderConfig()
    grounder.spatial_allocator = SpatialAllocatorConfig(spatial_policy="tpsa_boundary")
    sample = Sample(
        "sample", "video.mp4", "/video.mp4", 100.0, "a person waves",
        cardinality="multi",
    )
    instruction = grounder._prompt(sample, Component(0.0, 100.0, 1.0))[0]["content"][1]["text"]
    assert "every disjoint time interval" in instruction
    assert "do not omit repeated occurrences" in instruction
    assert "JSON array" in instruction


def test_full_video_baselines_share_the_timestamp_only_prompt():
    grounder = SemVIDGrounder.__new__(SemVIDGrounder)
    grounder.config = GrounderConfig()
    grounder.spatial_allocator = SpatialAllocatorConfig(spatial_policy="semvid")
    sample = Sample("sample", "video.mp4", "/video.mp4", 100.0, "open the drawer")
    instruction = grounder._prompt(sample, Component(0.0, 100.0, 1.0))[0]["content"][1]["text"]
    assert "Localize the event" in instruction
    assert '"present"' not in instruction


def test_neutral_per_frame_allocation_is_available_for_baselines():
    class Model:
        last_semantic_prune_coords = {
            "context": [torch.tensor([[0, 0], [1, 0]])],
            "object": [torch.tensor([[0, 1]])],
            "motion": [torch.tensor([[1, 2]])],
        }

    assert _per_frame_allocation(Model(), 0, 2, 4, {}, "semvid") == [2, 2]
    assert _per_frame_allocation(Model(), 0, 2, 4, {}, "dense") == [4, 4]
