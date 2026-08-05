from pathlib import Path

import pytest

from hybrid_vtg.semvid_bridge import validate_semvid_root


def test_missing_semvid_submodule_has_actionable_error(tmp_path: Path):
    with pytest.raises(RuntimeError, match="git submodule update"):
        validate_semvid_root(tmp_path)
