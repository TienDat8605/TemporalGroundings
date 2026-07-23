"""Deterministic utilities for hierarchical OMTG test-time search."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, float]:
        return {'start': round(self.start, 4), 'end': round(self.end, 4)}


def uniform_windows(duration: float, length: float = 45.0, overlap: float = 4.0) -> list[Window]:
    """Fallback windows with a 20-second final window whenever possible."""
    if duration <= length:
        return [Window(0.0, max(0.01, duration))]
    windows: list[Window] = []
    start = 0.0
    hop = max(1.0, length - overlap)
    while start < duration:
        end = min(duration, start + length)
        if duration - end < 20.0 and end < duration:
            end = duration
        windows.append(Window(start, end))
        if end >= duration:
            break
        start += hop
    return windows


def windows_from_boundaries(
    duration: float,
    boundaries: Iterable[float],
    *,
    min_seconds: float = 20.0,
    max_seconds: float = 60.0,
    overlap: float = 2.0,
) -> list[Window]:
    """Pack scene boundaries into bounded, slightly overlapping windows."""
    if duration <= min_seconds:
        return [Window(0.0, max(0.01, duration))]
    points = sorted({0.0, duration, *(
        min(duration, max(0.0, float(value))) for value in boundaries
    )})
    windows: list[Window] = []
    cursor = 0.0
    while cursor < duration - 1e-6:
        valid = [point for point in points if cursor + min_seconds <= point <= cursor + max_seconds]
        end = max(valid) if valid else min(duration, cursor + 45.0)
        if duration - end < min_seconds and end < duration:
            end = duration
        start = cursor if not windows else max(0.0, cursor - overlap)
        if end - start > max_seconds:
            start = end - max_seconds
        windows.append(Window(round(start, 4), round(end, 4)))
        if end >= duration:
            break
        cursor = end
    return windows


def content_aware_windows(video_path: str) -> tuple[list[Window], str]:
    """Use deterministic PySceneDetect boundaries, with a uniform fallback."""
    duration = probe_video(video_path)['duration']
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video, show_progress=False, frame_skip=4)
        scenes = manager.get_scene_list(start_in_scene=True)
        boundaries = [scene[1].get_seconds() for scene in scenes[:-1]]
        if not boundaries:
            raise RuntimeError('no content boundaries detected')
        return windows_from_boundaries(duration, boundaries), 'content'
    except Exception:
        return uniform_windows(duration), 'uniform-fallback'


def retained_window_count(number_of_windows: int) -> int:
    if number_of_windows <= 0:
        return 0
    return min(number_of_windows, min(8, max(2, math.ceil(math.sqrt(number_of_windows)))))


def uniform_window_indices(number_of_windows: int, count: int) -> list[int]:
    if count >= number_of_windows:
        return list(range(number_of_windows))
    return sorted(set(np.linspace(0, number_of_windows - 1, count).round().astype(int).tolist()))


def uniform_timestamps(start: float, end: float, count: int, phase: float = 0.5) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(start + end) / 2]
    width = (end - start) / count
    return [min(end - 1e-4, start + (index + phase) * width) for index in range(count)]


def distribute_frames(total: int, count: int, minimum: int = 2, factor: int = 2) -> list[int]:
    """Split frames without hidden Qwen video-frame padding."""
    if count <= 0:
        return []
    minimum_units = math.ceil(minimum / factor)
    total_units = max(math.ceil(total / factor), minimum_units * count)
    base, remainder = divmod(total_units, count)
    return [(base + int(index < remainder)) * factor for index in range(count)]


def probe_video(video_path: str) -> dict[str, float | int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f'Invalid video metadata: fps={fps}, frames={frames}, path={video_path}')
    return {'fps': fps, 'frames': frames, 'duration': frames / fps}


def extract_frames(
    video_path: str,
    timestamps: list[float],
    destination: Path,
    *,
    max_width: int = 336,
) -> list[str]:
    """Seek deterministic timestamps and cache JPEGs for multimodal inference."""
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / f'{index:04d}_{timestamp:.3f}.jpg' for index, timestamp in enumerate(timestamps)]
    missing = [(timestamp, path) for timestamp, path in zip(timestamps, paths) if not path.is_file()]
    if missing:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f'Unable to open video: {video_path}')
        for timestamp, path in missing:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f'Unable to decode {video_path} at {timestamp:.3f}s')
            height, width = frame.shape[:2]
            if width > max_width:
                ratio = max_width / width
                frame = cv2.resize(frame, (max_width, max(2, round(height * ratio))))
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                capture.release()
                raise RuntimeError(f'Unable to write frame: {path}')
        capture.release()
    return [str(path) for path in paths]


def interval_iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = first[1] - first[0] + second[1] - second[0] - intersection
    return intersection / union if union > 0 else 0.0


def consolidate_intervals(
    intervals: list[list[float]],
    *,
    duration: float,
    duplicate_iou: float = 0.8,
    merge_gap: float = 1.0,
) -> list[list[float]]:
    cleaned = sorted([
        [max(0.0, float(start)), min(duration, float(end))]
        for start, end in intervals
        if float(end) > float(start) and float(start) < duration and float(end) > 0
    ], key=lambda value: (value[0], value[1]))
    deduplicated: list[list[float]] = []
    for candidate in cleaned:
        duplicate = next((index for index, kept in enumerate(deduplicated)
                          if interval_iou(candidate, kept) >= duplicate_iou), None)
        if duplicate is None:
            deduplicated.append(candidate)
        elif candidate[1] - candidate[0] < deduplicated[duplicate][1] - deduplicated[duplicate][0]:
            deduplicated[duplicate] = candidate
    merged: list[list[float]] = []
    for start, end in sorted(deduplicated):
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[round(start, 3), round(end, 3)] for start, end in merged]


def router_recall(windows: list[Window], selected: list[int], targets: list[list[float]]) -> float:
    if not targets:
        return 1.0
    chosen = [windows[index] for index in selected]
    hits = sum(any(min(window.end, end) > max(window.start, start) for window in chosen)
               for start, end in targets)
    return hits / len(targets)


def oracle_window_indices(windows: list[Window], targets: list[list[float]], count: int) -> list[int]:
    scores = []
    for index, window in enumerate(windows):
        coverage = sum(max(0.0, min(window.end, end) - max(window.start, start)) for start, end in targets)
        scores.append((coverage, index))
    return [index for _, index in sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:count]]


def read_jsonl(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    records = {}
    if not path.is_file():
        return records
    with path.open('r+', encoding='utf-8') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A process can be killed between append and fsync. Repair only
                # an invalid final record; corruption in the middle is fatal.
                if handle.read():
                    raise
                handle.seek(offset)
                handle.truncate()
                break
            records[tuple(record[field] for field in key_fields)] = record
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class CallTimer:
    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        peak = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
        except Exception:
            pass
        self.wall_seconds = time.perf_counter() - self.started
        self.gpu_seconds = self.wall_seconds
        self.peak_vram_bytes = peak

    def to_dict(self) -> dict[str, float | int]:
        return {
            'wall_seconds': round(self.wall_seconds, 6),
            'gpu_seconds': round(self.gpu_seconds, 6),
            'peak_vram_bytes': self.peak_vram_bytes,
        }
