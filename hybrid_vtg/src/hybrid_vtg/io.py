"""Resumable JSONL output and reproducibility manifests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, mode: str = "write") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "w" if mode == "write" else "a"
    with path.open(file_mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default, ensure_ascii=False, sort_keys=True) + "\n")


def completed_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    """Return every attempted sample ID so append-only resumes never duplicate rows."""
    return {str(record["id"]) for record in records}


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_manifest(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value and not replace:
            raise RuntimeError(f"run manifest differs from existing output: {path}")
        if existing == value:
            return
    write_json(path, value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
