"""Canonical adapters for common VTG evaluation protocols."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .types import Sample
from .video import probe_video


VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi")
_CHARADES = re.compile(
    r"^(?P<video>\S+)\s+(?P<start>-?\d+(?:\.\d+)?)\s+(?P<end>-?\d+(?:\.\d+)?)##(?P<query>.+)$"
)


def _first_file(root: Path, candidates: tuple[str, ...]) -> Path:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of {candidates} exists under {root}")


def _video_index(root: Path) -> dict[str, Path]:
    directories = [path for path in (
        root / "videos", root / "rgb_videos_30fps_480", root / "rgb_videos_15fps_short256", root,
    ) if path.is_dir()]
    output: dict[str, Path] = {}
    for directory in directories:
        for suffix in VIDEO_SUFFIXES:
            for path in directory.rglob(f"*{suffix}"):
                output.setdefault(path.stem, path)
    if not output:
        raise FileNotFoundError(f"no videos found under {root}")
    return output


def load_charades_sta(root: Path, split: str = "test", maximum: int = 0) -> list[Sample]:
    annotation = _first_file(root, (
        f"charades_sta_{split}.txt", f"sta_annotation/charades_sta_{split}.txt",
        f"annotations/charades_sta_{split}.txt", f"annotations/{split}.txt",
    ))
    videos = _video_index(root)
    durations: dict[str, float] = {}
    rows = []
    for line_number, raw in enumerate(annotation.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        match = _CHARADES.match(raw.strip())
        if not match:
            raise ValueError(f"invalid Charades-STA line {line_number}: {raw!r}")
        video_id = Path(match.group("video")).stem
        if video_id not in videos:
            raise FileNotFoundError(f"missing Charades video {video_id}")
        if video_id not in durations:
            durations[video_id] = probe_video(videos[video_id]).duration
        duration = durations[video_id]
        start, end = float(match.group("start")), float(match.group("end"))
        sample = Sample(
            id=f"{split}:{line_number}", video=video_id, video_path=str(videos[video_id]), duration=duration,
            query=match.group("query").strip(), targets=((max(0.0, start), min(duration, end)),),
            group="charades-sta",
        )
        sample.validate()
        rows.append(sample)
        if maximum > 0 and len(rows) >= maximum:
            break
    return rows


def load_activitynet_grounding(root: Path, split: str = "val_2", maximum: int = 0) -> list[Sample]:
    annotation = _first_file(root, (
        f"{split}.json", f"captions/{split}.json", f"annotations/{split}.json",
        f"annotations/{split.replace('_', '')}.json",
    ))
    values = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("ActivityNet annotations must be a video-id mapping")
    videos = _video_index(root)
    rows = []
    for video_id, record in sorted(values.items()):
        clean_id = video_id.removeprefix("v_")
        path = videos.get(video_id) or videos.get(clean_id)
        if path is None:
            raise FileNotFoundError(f"missing ActivityNet video {video_id}")
        duration = float(record.get("duration") or probe_video(path).duration)
        timestamps, sentences = record.get("timestamps", []), record.get("sentences", [])
        if len(timestamps) != len(sentences):
            raise ValueError(f"annotation length mismatch for {video_id}")
        for event_index, (interval, sentence) in enumerate(zip(timestamps, sentences)):
            start, end = max(0.0, float(interval[0])), min(duration, float(interval[1]))
            if end <= start:
                continue
            sample = Sample(
                id=f"{video_id}:{event_index}", video=video_id, video_path=str(path), duration=duration,
                query=str(sentence).strip(), targets=((start, end),), group="activitynet-grounding",
            )
            sample.validate()
            rows.append(sample)
            if maximum > 0 and len(rows) >= maximum:
                return rows
    return rows


def load_jsonl(path: Path, maximum: int = 0) -> list[Sample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value: dict[str, Any] = json.loads(raw)
            video_path = str(value["video_path"])
            targets = tuple(tuple(map(float, interval)) for interval in value.get("targets", []))
            sample = Sample(
                id=str(value.get("id", line_number)), video=str(value.get("video", Path(video_path).stem)),
                video_path=video_path, duration=float(value.get("duration") or probe_video(video_path).duration),
                query=str(value["query"]), targets=targets, group=str(value.get("group", "custom")),
            )
            sample.validate()
            rows.append(sample)
            if maximum > 0 and len(rows) >= maximum:
                break
    return rows


def load_benchmark(name: str, source: Path, split: str, maximum: int = 0) -> list[Sample]:
    if name == "charades-sta":
        return load_charades_sta(source, split, maximum)
    if name in {"activitynet-grounding", "activitynet-captions"}:
        return load_activitynet_grounding(source, split, maximum)
    if name == "jsonl":
        return load_jsonl(source, maximum)
    raise ValueError(f"unknown benchmark {name!r}")
