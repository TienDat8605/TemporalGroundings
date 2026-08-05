"""Read-only local runtime validation."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from .semvid_bridge import validate_semvid_root


def inspect_runtime(semvid_root: Path) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    try:
        validate_semvid_root(semvid_root)
        checks.append(("SemVID submodule", True, str(semvid_root)))
    except RuntimeError as error:
        checks.append(("SemVID submodule", False, str(error)))
    for module in ("torch", "transformers", "decord", "qwen_vl_utils", "PIL", "numpy"):
        available = importlib.util.find_spec(module) is not None
        checks.append((f"Python module {module}", available, "installed" if available else "missing"))
    checks.append(("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "missing"))
    if importlib.util.find_spec("torch") is not None:
        import torch
        cuda = torch.cuda.is_available()
        detail = "not available"
        if cuda:
            detail = ", ".join(
                f"{torch.cuda.get_device_name(index)} ({torch.cuda.get_device_properties(index).total_memory / 2**30:.1f} GiB)"
                for index in range(torch.cuda.device_count())
            )
        checks.append(("CUDA", cuda, detail))
    return checks
