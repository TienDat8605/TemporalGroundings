"""Canonical dataset adapters for training-free VTG evaluation."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CanonicalRow:
    id: str
    video: str
    video_path: str
    duration: float
    query: str
    targets: tuple[tuple[float, float], ...]
    group: str
    cardinality: str
    modalities: tuple[str, ...]
    native_protocol: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['targets'] = [list(interval) for interval in self.targets]
        value['modalities'] = list(self.modalities)
        return value


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    root_env: str
    cardinality: str
    modalities: tuple[str, ...]
    native_protocol: str
    metric_family: str
    loader: Callable[[Path, str, int], tuple[list[CanonicalRow], dict[str, Any]]]


def _video_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'missing video directory: {directory}')
    output = {}
    for suffix in ('*.mp4', '*.mkv', '*.webm', '*.avi'):
        for path in sorted(directory.rglob(suffix)):
            output.setdefault(path.stem, path)
    return output


def _duration(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'unable to open video: {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f'invalid video metadata: {path}')
    return frames / fps


def _limit(rows: list[CanonicalRow], maximum: int) -> list[CanonicalRow]:
    return rows[:maximum] if maximum > 0 else rows


_CHARADES_LINE = re.compile(
    r'^(?P<video>\S+)\s+(?P<start>-?\d+(?:\.\d+)?)\s+(?P<end>-?\d+(?:\.\d+)?)##(?P<query>.+)$'
)


def load_charades_sta(
    root: Path,
    split: str = 'test',
    maximum: int = 0,
) -> tuple[list[CanonicalRow], dict[str, Any]]:
    annotations = next((path for path in (
        root / f'charades_sta_{split}.txt',
        root / 'annotations' / f'charades_sta_{split}.txt',
        root / 'annotations' / f'{split}.txt',
    ) if path.is_file()), None)
    if annotations is None:
        raise FileNotFoundError(f'missing Charades-STA {split} annotations under {root}')
    videos = _video_index(root / 'videos')
    rows = []
    with annotations.open(encoding='utf-8') as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            match = _CHARADES_LINE.match(line)
            if not match:
                raise ValueError(f'invalid Charades-STA line {line_number}: {line!r}')
            video_id = match.group('video')
            video_path = videos.get(Path(video_id).stem)
            if video_path is None:
                raise FileNotFoundError(f'missing Charades video: {video_id}')
            start, end = float(match.group('start')), float(match.group('end'))
            duration = _duration(video_path)
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f'invalid Charades interval at line {line_number}')
            rows.append(CanonicalRow(
                id=f'{split}:{line_number}',
                video=video_id,
                video_path=str(video_path),
                duration=duration,
                query=match.group('query').strip(),
                targets=((max(0.0, start), min(duration, end)),),
                group='charades-sta',
                cardinality='single',
                modalities=('video',),
                native_protocol='continuous-seconds',
            ))
    rows = _limit(rows, maximum)
    return rows, {'annotation': str(annotations), 'split': split, 'samples': len(rows)}


def load_activitynet_captions(
    root: Path,
    split: str = 'val_1',
    maximum: int = 0,
) -> tuple[list[CanonicalRow], dict[str, Any]]:
    annotations = next((path for path in (
        root / f'{split}.json',
        root / 'annotations' / f'{split}.json',
        root / 'annotations' / f'{split.replace("_", "")}.json',
    ) if path.is_file()), None)
    if annotations is None:
        raise FileNotFoundError(f'missing ActivityNet Captions {split} annotations under {root}')
    data = json.loads(annotations.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('ActivityNet Captions annotations must be a video-id mapping')
    videos = _video_index(root / 'videos')
    rows = []
    for video_id in sorted(data):
        record = data[video_id]
        clean_id = video_id[2:] if video_id.startswith('v_') else video_id
        video_path = videos.get(video_id) or videos.get(clean_id)
        if video_path is None:
            raise FileNotFoundError(f'missing ActivityNet video: {video_id}')
        duration = float(record.get('duration') or _duration(video_path))
        timestamps = record.get('timestamps') or []
        sentences = record.get('sentences') or []
        if len(timestamps) != len(sentences):
            raise ValueError(f'ActivityNet annotation length mismatch: {video_id}')
        for event_index, (interval, sentence) in enumerate(zip(timestamps, sentences)):
            start, end = float(interval[0]), float(interval[1])
            if not (0 <= start < end and math.isfinite(start) and math.isfinite(end)):
                continue
            rows.append(CanonicalRow(
                id=f'{video_id}:{event_index}',
                video=video_id,
                video_path=str(video_path),
                duration=duration,
                query=str(sentence).strip(),
                targets=((max(0.0, start), min(duration, end)),),
                group='activitynet-captions',
                cardinality='single',
                modalities=('video',),
                native_protocol='continuous-seconds',
            ))
    rows = _limit(rows, maximum)
    return rows, {'annotation': str(annotations), 'split': split, 'samples': len(rows)}


ADAPTERS = {
    'charades-sta': AdapterSpec(
        name='charades-sta',
        root_env='CHARADES_STA_ROOT',
        cardinality='single',
        modalities=('video',),
        native_protocol='continuous-seconds',
        metric_family='r1-iou-0.3-0.5-0.7-miou',
        loader=load_charades_sta,
    ),
    'activitynet-captions': AdapterSpec(
        name='activitynet-captions',
        root_env='ACTIVITYNET_CAPTIONS_ROOT',
        cardinality='single',
        modalities=('video',),
        native_protocol='continuous-seconds',
        metric_family='r1-iou-0.3-0.5-0.7-miou',
        loader=load_activitynet_captions,
    ),
}


def load_adapter(
    name: str,
    root: Path,
    *,
    split: str,
    maximum: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        spec = ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f'unknown standalone adapter: {name}') from error
    rows, metadata = spec.loader(root, split, maximum)
    metadata.update({
        'dataset': name,
        'cardinality': spec.cardinality,
        'modalities': list(spec.modalities),
        'native_protocol': spec.native_protocol,
        'metric_family': spec.metric_family,
    })
    return [row.to_dict() for row in rows], metadata


def validate_canonical_rows(rows: list[dict[str, Any]]) -> None:
    required = {
        'id', 'video', 'video_path', 'duration', 'query', 'targets',
        'group', 'cardinality', 'modalities', 'native_protocol',
    }
    seen = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f'canonical row is missing fields: {sorted(missing)}')
        if row['id'] in seen:
            raise ValueError(f'duplicate row id: {row["id"]}')
        seen.add(row['id'])
        if row['cardinality'] not in ('single', 'multi'):
            raise ValueError(f'invalid cardinality: {row["cardinality"]}')
        if float(row['duration']) <= 0 or not str(row['query']).strip():
            raise ValueError(f'invalid canonical row: {row["id"]}')
        for start, end in row['targets']:
            if not 0 <= float(start) < float(end) <= float(row['duration']) + 1e-3:
                raise ValueError(f'invalid target for row: {row["id"]}')
