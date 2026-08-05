from pathlib import Path

import pytest

from hybrid_vtg.semvid_bridge import _render_generation_prompt, validate_semvid_root


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
