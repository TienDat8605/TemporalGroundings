"""Canonical adapters for common VTG evaluation protocols."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .timestamps import parse_intervals
from .types import Sample
from .video import probe_video


VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi")
_CHARADES = re.compile(
    r"^(?P<video>\S+)\s+(?P<start>-?\d+(?:\.\d+)?)\s+(?P<end>-?\d+(?:\.\d+)?)##(?P<query>.+)$"
)
_OMTG_QUERY = re.compile(
    r"textual query\s+['\"](?P<query>.+)['\"]\s+and determine",
    re.IGNORECASE,
)


def _first_file(root: Path, candidates: tuple[str, ...]) -> Path:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of {candidates} exists under {root}")


def _video_index(root: Path) -> dict[str, Path]:
    directories = [path for path in (
        root / "videos", root / "rgb_videos_30fps_480", root / "rgb_videos_15fps_short256",
    ) if path.is_dir()]
    if not directories:
        directories = [root]
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


def load_tacos(root: Path, split: str = "test", maximum: int = 0) -> list[Sample]:
    annotation = _first_file(root, (
        f"{split}.jsonl", f"annotations/{split}.jsonl", f"captions/{split}.jsonl",
    ))
    videos = _video_index(root)
    rows = []
    with annotation.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            record: dict[str, Any] = json.loads(raw)
            video_id = str(record["vid"])
            path = videos.get(video_id)
            if path is None:
                raise FileNotFoundError(f"missing TACoS video {video_id}")
            duration = float(record.get("duration") or probe_video(path).duration)
            targets = tuple(
                (max(0.0, float(interval[0])), min(duration, float(interval[1])))
                for interval in record.get("relevant_windows", [])
                if len(interval) == 2 and float(interval[1]) > float(interval[0])
            )
            sample = Sample(
                id=str(record.get("qid", f"{split}:{line_number}")), video=video_id,
                video_path=str(path), duration=duration, query=str(record["query"]).strip(),
                targets=targets, group="tacos",
            )
            sample.validate()
            rows.append(sample)
            if maximum > 0 and len(rows) >= maximum:
                break
    return rows


def load_omtg(root: Path, split: str = "test", maximum: int = 0) -> list[Sample]:
    """Load the fixed 320-query OMTG Bench TSV using its native multi-span labels."""
    if split not in {"test", "all"}:
        raise ValueError("OMTG Bench has one fixed evaluation split; use 'test'")
    annotation = _first_file(root, ("OMTGBench.tsv",))
    videos = _video_index(root)
    durations: dict[str, float] = {}
    rows = []
    with annotation.open(encoding="utf-8", newline="") as handle:
        for line_number, record in enumerate(csv.DictReader(handle, delimiter="\t"), 1):
            video_name = str(record["video"])
            video_id = Path(video_name).stem
            path = videos.get(video_id)
            if path is None:
                raise FileNotFoundError(f"missing OMTG video {video_name}")
            question = str(record["question"]).strip()
            match = _OMTG_QUERY.search(question)
            query = match.group("query").strip() if match else question
            targets = parse_intervals(str(record["answer"]))
            if video_id not in durations:
                durations[video_id] = probe_video(path).duration
            labelled_end = max((end for _, end in targets), default=0.0)
            durations[video_id] = max(durations[video_id], labelled_end)
            sample = Sample(
                id=str(record.get("id", line_number - 1)),
                video=video_name,
                video_path=str(path),
                duration=durations[video_id],
                query=query,
                targets=targets,
                group="omtg",
                cardinality="multi",
            )
            sample.validate()
            rows.append(sample)
            if maximum > 0 and len(rows) >= maximum:
                break
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
                cardinality=str(value.get("cardinality", "single")),
            )
            sample.validate()
            rows.append(sample)
            if maximum > 0 and len(rows) >= maximum:
                break
    return rows


def load_benchmark(name: str, source: Path, split: str, maximum: int = 0) -> list[Sample]:
    if name == "omtg":
        return load_omtg(source, split, maximum)
    if name == "charades-sta":
        return load_charades_sta(source, split, maximum)
    if name in {"activitynet-grounding", "activitynet-captions"}:
        return load_activitynet_grounding(source, split, maximum)
    if name == "tacos":
        return load_tacos(source, split, maximum)
    if name == "jsonl":
        return load_jsonl(source, maximum)
    raise ValueError(f"unknown benchmark {name!r}")
