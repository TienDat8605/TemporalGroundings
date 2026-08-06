from pathlib import Path

import pytest

from hybrid_vtg.config import SemVIDConfig
from hybrid_vtg.semvid_bridge import SemVIDGrounder, _render_generation_prompt, validate_semvid_root
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


def test_grounding_prompt_allows_abstention_and_requires_absolute_timestamps():
    grounder = SemVIDGrounder.__new__(SemVIDGrounder)
    grounder.config = SemVIDConfig()
    sample = Sample("sample", "video.mp4", "/video.mp4", 100.0, "open the drawer")
    prompt = grounder._prompt(sample, Component(20.0, 40.0, 0.5))
    instruction = prompt[0]["content"][1]["text"]
    assert '"present": false' in instruction
    assert '"present": true' in instruction
    assert "original-video timestamps, not clip-relative time" in instruction
