"""Shared benchmark file discovery."""

from __future__ import annotations

from pathlib import Path

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi")


def first_file(root: Path, candidates: tuple[str, ...]) -> Path:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of {candidates} exists under {root}")


def video_index(root: Path) -> dict[str, Path]:
    directories = [
        path
        for path in (
            root / "videos",
            root / "rgb_videos_30fps_480",
            root / "rgb_videos_15fps_short256",
        )
        if path.is_dir()
    ] or [root]
    output: dict[str, Path] = {}
    for directory in directories:
        for suffix in VIDEO_SUFFIXES:
            for path in directory.rglob(f"*{suffix}"):
                output.setdefault(path.stem, path)
    if not output:
        raise FileNotFoundError(f"no videos found under {root}")
    return output
