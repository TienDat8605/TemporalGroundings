#!/usr/bin/env python3
"""Run strict-budget hierarchical TimeLens2 search across VTG benchmarks."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import gc
import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from vlmeval.dataset.omtgbench import parse_time_intervals
from vlmeval.dataset.vidi_vue_tr import (
    _compute_precision_recall,
    _overlap_ratio,
    _success_overlap,
)
from vlmeval.omtg_search import (
    CallTimer,
    Window,
    append_jsonl,
    consolidate_intervals,
    content_aware_windows,
    distribute_frames,
    estimated_sampled_frames,
    extract_frames,
    probe_video,
    read_jsonl,
    router_recall,
    strict_embedding_policy,
    uniform_timestamps,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT.parent / 'results' / 'vtg_search'
SCHEDULES = ('uniform-one-shot', 'embedding-window-local')
PROMPT_MODES = ('controlled', 'native-style')
NATIVE_SCHEDULE = 'native-fps-capped'
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    root_env: str
    default_root: str
    native_fps: float
    cardinality: str
    native_prompt: str
    thresholds: tuple[float, ...]
    grouping_field: str

    @property
    def multi_span(self) -> bool:
        return self.cardinality == 'multi'


BENCHMARKS = {
    'vue-tr-v2': BenchmarkSpec(
        name='vue-tr-v2',
        root_env='VUE_TR_V2_ROOT',
        default_root=str(REPOSITORY_ROOT / 'data' / 'VUE_TR_V2'),
        native_fps=1.0,
        cardinality='multi',
        native_prompt=(
            'You are given a video.\nTask: temporal retrieval.\n'
            'Given the query: "{query}", return ALL time spans (in seconds) where the query is relevant.\n'
            'Output format MUST be a JSON array of [start, end] pairs, '
            'e.g. [[0, 3.5], [10, 12]].'
        ),
        thresholds=(0.3, 0.5, 0.7),
        grouping_field='duration_category',
    ),
    'momentseeker': BenchmarkSpec(
        name='momentseeker',
        root_env='MOMENT_SEEKER_ROOT',
        default_root=str(REPOSITORY_ROOT / 'data' / 'MomentSeeker'),
        native_fps=2.0,
        cardinality='multi',
        native_prompt=(
            'You are given a video.\nTask: temporal grounding.\n'
            'Given the query: "{query}", return ALL time spans (in seconds) where the query '
            'is grounded in the video.\nOutput format MUST be a JSON array of [start, end] '
            'pairs, e.g. [[0, 3.5], [10, 12]].'
        ),
        thresholds=(0.3, 0.5, 0.7),
        grouping_field='task',
    ),
    'ego4d-nlq-v2': BenchmarkSpec(
        name='ego4d-nlq-v2',
        root_env='EGO4D_NLQ_V2_ROOT',
        default_root=str(REPOSITORY_ROOT / 'data' / 'Ego4D-NLQ-v2'),
        native_fps=2.0,
        cardinality='single',
        native_prompt=(
            'You are given a video.\nTask: temporal grounding.\n'
            'Given the query: "{query}", return the time span (in seconds) where the query '
            'is grounded in the video.\nOutput format MUST be [[start, end]].'
        ),
        thresholds=(0.1, 0.2, 0.3, 0.5, 0.7),
        grouping_field='query_type',
    ),
    'qvhighlights-timelens': BenchmarkSpec(
        name='qvhighlights-timelens',
        root_env='TIMELENS_BENCH_ROOT',
        default_root=str(REPOSITORY_ROOT / 'data' / 'TimeLens-Bench'),
        native_fps=4.0,
        cardinality='single',
        native_prompt=(
            "Please find the visual event described by the sentence '{query}', determining "
            "its starting and ending times. The format should be: "
            "'The event happens in <start time> - <end time> seconds'."
        ),
        thresholds=(0.3, 0.5, 0.7),
        grouping_field='subset',
    ),
}


def parse_csv(value: str, converter=str) -> list:
    return [converter(item.strip()) for item in value.split(',') if item.strip()]


def _structured(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return []


def _intervals(value: Any) -> list[list[float]]:
    parsed = _structured(value)
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2 and all(
        isinstance(item, (int, float)) for item in parsed
    ):
        parsed = [parsed]
    output = []
    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                start, end = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and end > start:
                output.append([start, end])
    return output


def _video_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing video directory: {directory}')
    output: dict[str, Path] = {}
    for path in sorted(directory.glob('*.mp4')):
        key = path.name.split('.')[0]
        current = output.get(key)
        if current is None or (len(path.name), path.name) < (len(current.name), current.name):
            output[key] = path
    return output


def _read_table(path: Path) -> list[dict]:
    return pd.read_csv(path, sep='\t').to_dict(orient='records')


def _canonical_row(
    *,
    sample_id: Any,
    video_id: Any,
    video_path: Path,
    duration: Any,
    query: Any,
    targets: Any,
    group: Any = '',
) -> dict:
    target_intervals = _intervals(targets)
    if not target_intervals:
        raise ValueError(f'Sample {sample_id!r} has no valid target intervals')
    actual_duration = float(duration or 0)
    if actual_duration <= 0:
        actual_duration = float(probe_video(str(video_path))['duration'])
    return {
        'id': str(sample_id),
        'video': str(video_id),
        'video_path': str(video_path.resolve()),
        'duration': actual_duration,
        'query': str(query).strip(),
        'targets': target_intervals,
        'group': str(group or ''),
    }


def _load_vue(root: Path) -> tuple[list[dict], dict]:
    videos = _video_index(root / 'videos')
    tsv = root / 'VUE_TR_V2.tsv'
    official_annotation = root / 'VUE-TRv2_ground_truth.json'
    annotation = tsv if tsv.is_file() else official_annotation
    source_rows = _read_table(tsv) if tsv.is_file() else json.loads(annotation.read_text(encoding='utf-8'))
    rows = []
    for position, item in enumerate(source_rows):
        if str(item.get('query_modality', 'vision')).strip().lower() != 'vision':
            continue
        video_id = str(item.get('video_id', item.get('video', '')))
        if video_id not in videos:
            continue
        rows.append(_canonical_row(
            sample_id=item.get('query_id', item.get('index', position)),
            video_id=video_id,
            video_path=videos[video_id],
            duration=item.get('duration', 0),
            query=item.get('query', item.get('question', '')),
            targets=item.get('gt'),
            group=item.get('duration_category', ''),
        ))
    expected_source = (
        json.loads(official_annotation.read_text(encoding='utf-8'))
        if official_annotation.is_file() else source_rows
    )
    expected_rows = sum(
        str(item.get('query_modality', 'vision')).strip().lower() == 'vision'
        for item in expected_source
    )
    return rows, {'annotation': str(annotation), 'expected_rows': expected_rows}


def _load_momentseeker(root: Path) -> tuple[list[dict], dict]:
    videos = _video_index(root / 'videos')
    tsv = root / 'MomentSeeker.tsv'
    annotation = tsv if tsv.is_file() else root / 't2v.json'
    source_rows = _read_table(tsv) if tsv.is_file() else json.loads(annotation.read_text(encoding='utf-8'))
    rows = []
    for position, item in enumerate(source_rows):
        if tsv.is_file():
            video_id = str(item.get('video_id', item.get('video', '')))
            query = item.get('query', item.get('question', ''))
            targets = item.get('gt')
        else:
            video_id = Path(str(item.get('src_video_path', ''))).stem
            query = item.get('qry_text', '')
            targets = item.get('answering_time_interval')
        if video_id not in videos or not _intervals(targets):
            continue
        rows.append(_canonical_row(
            sample_id=item.get('index', position),
            video_id=video_id,
            video_path=videos[video_id],
            duration=item.get('duration', 0),
            query=query,
            targets=targets,
            group=item.get('task', ''),
        ))
    return rows, {'annotation': str(annotation), 'expected_rows': len(source_rows)}


def _load_ego4d(root: Path) -> tuple[list[dict], dict]:
    videos_root = Path(os.environ.get('EGO4D_NLQ_V2_VIDEOS_DIR', root / 'videos'))
    videos = _video_index(videos_root)
    tsv = root / 'Ego4D-NLQ-v2_val.tsv'
    flat_jsonl = root / 'ego4d_nlq_val_v2.jsonl'
    official_json = root / 'nlq_val.json'
    annotation = tsv if tsv.is_file() else (
        flat_jsonl if flat_jsonl.is_file() else official_json
    )
    if tsv.is_file():
        source_rows = _read_table(tsv)
    elif annotation.suffix == '.jsonl':
        source_rows = [
            json.loads(line) for line in annotation.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(annotation.read_text(encoding='utf-8'))
        source_rows = []
        for video in payload.get('videos', []):
            video_id = str(video.get('video_uid', ''))
            for clip in video.get('clips', []):
                clip_id = str(clip.get('clip_uid', ''))
                clip_duration = max(
                    0.0,
                    float(clip.get('clip_end_sec', 0)) - float(clip.get('clip_start_sec', 0)),
                )
                for annotation_item in clip.get('annotations', []):
                    annotation_id = str(annotation_item.get('annotation_uid', ''))
                    for query_index, language_query in enumerate(
                        annotation_item.get('language_queries', [])
                    ):
                        query = language_query.get('query')
                        if not query:
                            continue
                        if clip_id in videos:
                            row_video_id = clip_id
                            timestamps = [[
                                language_query.get('clip_start_sec'),
                                language_query.get('clip_end_sec'),
                            ]]
                            duration = clip_duration
                        else:
                            row_video_id = video_id
                            timestamps = [[
                                language_query.get('video_start_sec'),
                                language_query.get('video_end_sec'),
                            ]]
                            duration = 0.0
                        source_rows.append({
                            'video_id': row_video_id,
                            'query_id': f'{annotation_id}:{query_index}',
                            'query': query,
                            'duration': duration,
                            'timestamps': timestamps,
                            'query_type': str(language_query.get('template') or 'nlq'),
                        })
    rows = []
    for position, item in enumerate(source_rows):
        video_id = str(item.get('video_id', item.get('video', '')))
        if video_id not in videos:
            continue
        rows.append(_canonical_row(
            sample_id=item.get('query_id', item.get('index', position)),
            video_id=video_id,
            video_path=videos[video_id],
            duration=item.get('duration', 0),
            query=re.sub(
                r'^query\s*text\s*:\s*', '', str(item.get('query', item.get('question', ''))),
                flags=re.IGNORECASE,
            ),
            targets=item.get('timestamps', item.get('gt')),
            group=item.get('query_type', 'nlq'),
        ))
    return rows, {'annotation': str(annotation), 'expected_rows': len(source_rows)}


def _load_qvhighlights(root: Path) -> tuple[list[dict], dict]:
    annotation = root / 'qvhighlights-timelens.json'
    videos_root = root / 'videos' / 'qvhighlights'
    data = json.loads(annotation.read_text(encoding='utf-8'))
    rows = []
    position = 0
    for video_id, item in data.items():
        path = videos_root / f'{video_id}.mp4'
        if not path.is_file():
            continue
        for span, query in zip(item.get('spans', []), item.get('queries', [])):
            rows.append(_canonical_row(
                sample_id=position,
                video_id=video_id,
                video_path=path,
                duration=item.get('duration', 0),
                query=re.sub(r'\s+', ' ', str(query)).strip().strip('.'),
                targets=[span],
                group='qvhighlights',
            ))
            position += 1
    expected = sum(min(len(item.get('spans', [])), len(item.get('queries', []))) for item in data.values())
    return rows, {'annotation': str(annotation), 'expected_rows': expected}


LOADERS: dict[str, Callable[[Path], tuple[list[dict], dict]]] = {
    'vue-tr-v2': _load_vue,
    'momentseeker': _load_momentseeker,
    'ego4d-nlq-v2': _load_ego4d,
    'qvhighlights-timelens': _load_qvhighlights,
}


def load_benchmark(spec: BenchmarkSpec, root: Path, max_samples: int) -> tuple[list[dict], dict]:
    rows, metadata = LOADERS[spec.name](root)
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{spec.name} contains duplicate sample ids')
    metadata['available_rows'] = len(rows)
    metadata['coverage'] = len(rows) / max(1, int(metadata['expected_rows']))
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows, metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_dir(cache_root: Path, row: dict, label: str) -> Path:
    safe_video = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(row['video']).stem)
    safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', row['id'])
    safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
    return cache_root / safe_video / safe_id / safe_label


def build_prompt(
    spec: BenchmarkSpec,
    query: str,
    duration: float,
    *,
    local: bool,
    mode: str,
) -> str:
    coordinate = 'relative to the start of this clip' if local else 'relative to the full video'
    if mode == 'controlled':
        cardinality = (
            'Find every disjoint time interval where this event occurs'
            if spec.multi_span else
            'Find the single best time interval where this event occurs'
        )
        empty = ' Return [] when the event does not occur.' if spec.multi_span else ''
        return (
            f'{cardinality}: {query!r}. Timestamps must be in seconds {coordinate}, '
            f'between 0 and {duration:.3f}. Return only a JSON array of [start, end] '
            f'pairs, ordered by start time.{empty}'
        )
    if mode != 'native-style':
        raise ValueError(f'Unknown prompt mode: {mode}')
    return (
        spec.native_prompt.format(query=query).rstrip()
        + f'\nUse seconds {coordinate}, between 0 and {duration:.3f}.'
    )


def timed_ground_call(model, frames: list[str], prompt: str, duration: float) -> tuple[str, list[list[float]], dict]:
    message = [
        {
            'type': 'video',
            'value': frames,
            'sample_fps': max(0.01, len(frames) / max(duration, 1e-3)),
        },
        {'type': 'text', 'value': prompt},
    ]
    with CallTimer() as timer:
        response = model.generate(message, dataset='VTGSearch')
    return response, parse_time_intervals(response), timer.to_dict()


def _windows_for_route(video_path: Path) -> tuple[list[Window], str]:
    return content_aware_windows(str(video_path))


def route_phase(args, spec: BenchmarkSpec, rows: list[dict], run_dir: Path, cache_root: Path) -> None:
    import torch
    from vlmeval.vlm.qwen3_vl_embedding import Qwen3VLEmbedder

    route_path = run_dir / 'routes.jsonl'
    complete = read_jsonl(route_path, ('id', 'budget'))
    pending = [
        (row, budget) for row in rows for budget in args.budgets
        if (row['id'], budget) not in complete
    ]
    if not pending:
        print(f'Routing already complete: {len(complete)} records')
        return
    print(f'Loading router {args.embedding_model} for {len(pending)} routes...')
    embedder = Qwen3VLEmbedder(
        args.embedding_model,
        max_pixels=args.embedding_max_pixels,
        total_pixels=4 * args.embedding_max_pixels,
        attn_implementation=args.attention,
    )
    window_cache: dict[str, dict] = {}
    cache_path = run_dir / 'video_windows.json'
    if cache_path.is_file():
        window_cache = json.loads(cache_path.read_text(encoding='utf-8'))

    for position, (row, budget) in enumerate(pending, 1):
        video_path = Path(row['video_path'])
        started = time.perf_counter()
        if row['video'] not in window_cache:
            windows, source = _windows_for_route(video_path)
            window_cache[row['video']] = {
                'source': source,
                'windows': [window.to_dict() for window in windows],
            }
            temporary = cache_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(window_cache, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            temporary.replace(cache_path)
        original = [Window(**value) for value in window_cache[row['video']]['windows']]
        windows, policy = strict_embedding_policy(original, budget)
        windowing_seconds = time.perf_counter() - started

        if len(windows) == 1:
            append_jsonl(route_path, {
                'id': row['id'], 'budget': budget, 'video': row['video'],
                'duration': row['duration'], 'window_source': window_cache[row['video']]['source'],
                'windows': [windows[0].to_dict()], 'scores': [1.0], 'selected': [0],
                'bypass': True, **policy,
                'telemetry': {
                    'wall_seconds': round(windowing_seconds, 6), 'gpu_seconds': 0.0,
                    'peak_vram_bytes': 0, 'embedding_calls': 0,
                },
            })
            continue

        timers = []
        route_started = time.perf_counter()
        with CallTimer() as query_timer:
            query_embedding = embedder.process([{
                'text': row['query'],
                'instruction': 'Represent this text for retrieving matching temporal video windows.',
            }])[0]
        timers.append(query_timer.to_dict())
        scores = []
        for index, window in enumerate(windows):
            count = policy['router_frames_per_window']
            frames = extract_frames(
                str(video_path),
                uniform_timestamps(window.start, window.end, count),
                frame_dir(cache_root, row, f'route_{budget}_{index}_{count}f'),
                max_width=args.frame_width,
            )
            with CallTimer() as timer:
                embedding = embedder.process([{
                    'video': frames,
                    'sample_fps': count / max(window.duration, 1e-3),
                    'instruction': 'Represent this video window for retrieval by a textual event description.',
                }])[0]
            timers.append(timer.to_dict())
            scores.append(float(torch.dot(query_embedding.float(), embedding.float()).item()))
        selected = sorted(range(len(windows)), key=lambda index: (-scores[index], index))[
            :policy['selected_window_count']
        ]
        append_jsonl(route_path, {
            'id': row['id'], 'budget': budget, 'video': row['video'],
            'duration': row['duration'], 'window_source': window_cache[row['video']]['source'],
            'windows': [window.to_dict() for window in windows],
            'scores': scores, 'selected': selected, 'bypass': False, **policy,
            'router_recall': router_recall(windows, selected, row['targets']),
            'telemetry': {
                'wall_seconds': round(windowing_seconds + time.perf_counter() - route_started, 6),
                'gpu_seconds': round(sum(value['gpu_seconds'] for value in timers), 6),
                'peak_vram_bytes': max(value['peak_vram_bytes'] for value in timers),
                'embedding_calls': len(timers),
            },
        })
        print(f'[route {position}/{len(pending)}] id={row["id"]} budget={budget} windows={len(windows)}')

    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_setting(
    *,
    model,
    spec: BenchmarkSpec,
    row: dict,
    schedule: str,
    prompt_mode: str,
    budget: int,
    route: dict | None,
    cache_root: Path,
    frame_width: int,
    native_frame_cap: int,
) -> dict:
    duration = float(row['duration'])
    video_path = Path(row['video_path'])
    calls = []
    candidates = []
    grounder_frames = 0
    ground_started = time.perf_counter()

    def call_window(window: Window, count: int, label: str, score: float, local: bool):
        nonlocal grounder_frames
        frames = extract_frames(
            str(video_path),
            uniform_timestamps(window.start, window.end, count),
            frame_dir(cache_root, row, label),
            max_width=frame_width,
        )
        response, intervals, telemetry = timed_ground_call(
            model,
            frames,
            build_prompt(
                spec, row['query'], window.duration if local else duration,
                local=local, mode=prompt_mode,
            ),
            window.duration if local else duration,
        )
        valid = []
        for interval_order, (start, end) in enumerate(intervals):
            if local:
                start = max(0.0, start)
                end = min(window.duration, end)
                if end <= start:
                    continue
                interval = [start + window.start, end + window.start]
            else:
                interval = [max(0.0, start), min(duration, end)]
                if interval[1] <= interval[0]:
                    continue
            valid.append(interval)
            candidates.append({
                'interval': interval,
                'router_score': score,
                'call_order': len(calls),
                'interval_order': interval_order,
            })
        grounder_frames += len(frames)
        calls.append({
            'window': window.to_dict(), 'router_score': score,
            'frame_count': len(frames), 'response': response,
            'parsed_intervals': valid, **telemetry,
        })

    if schedule == 'uniform-one-shot':
        call_window(Window(0.0, duration), budget, f'uniform_{budget}', 1.0, False)
        embedding_frames = 0
        route_telemetry = {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'peak_vram_bytes': 0}
        bypass = False
    elif schedule == 'embedding-window-local':
        if route is None:
            raise ValueError('embedding-window-local requires a route')
        route_telemetry = route['telemetry']
        bypass = bool(route['bypass'])
        if bypass:
            call_window(Window(0.0, duration), budget, f'embedding_bypass_{budget}', 1.0, False)
            embedding_frames = 0
        else:
            windows = [Window(**value) for value in route['windows']]
            selected = [int(index) for index in route['selected']]
            allocations = distribute_frames(int(route['local_budget']), len(selected))
            for index, allocation in zip(selected, allocations):
                call_window(
                    windows[index], allocation, f'embedding_{budget}_{index}',
                    float(route['scores'][index]), True,
                )
            embedding_frames = int(route['router_frames'])
    elif schedule == NATIVE_SCHEDULE:
        requested = estimated_sampled_frames(duration, spec.native_fps)
        count = min(native_frame_cap, requested)
        if count % 2:
            count = max(2, count - 1)
        call_window(Window(0.0, duration), count, f'native_{spec.native_fps:g}fps_{count}', 1.0, False)
        embedding_frames = 0
        route_telemetry = {'gpu_seconds': 0.0, 'wall_seconds': 0.0, 'peak_vram_bytes': 0}
        bypass = False
    else:
        raise ValueError(f'Unknown schedule: {schedule}')

    if spec.multi_span:
        prediction = consolidate_intervals(
            [value['interval'] for value in candidates], duration=duration
        )
    elif candidates:
        best = min(
            candidates,
            key=lambda value: (
                -value['router_score'], value['call_order'], value['interval_order']
            ),
        )
        prediction = [[round(best['interval'][0], 3), round(best['interval'][1], 3)]]
    else:
        prediction = []
    total_frames = grounder_frames + embedding_frames
    if schedule in SCHEDULES and total_frames > budget:
        raise AssertionError(f'{schedule} exceeded budget: {total_frames}>{budget}')
    ground_wall = time.perf_counter() - ground_started
    router_calls = int(route_telemetry.get('embedding_calls', 0))
    return {
        'prediction': prediction,
        'calls': calls,
        'grounder_calls': len(calls),
        'router_calls': router_calls,
        'model_calls': len(calls) + router_calls,
        'bypass': bypass,
        'grounder_frames': grounder_frames,
        'embedding_frames': embedding_frames,
        'total_frames': total_frames,
        'unused_frames': max(0, budget - total_frames) if schedule in SCHEDULES else 0,
        'budget_overflow_frames': max(0, total_frames - budget) if schedule in SCHEDULES else 0,
        'grounder_model_seconds': sum(call['gpu_seconds'] for call in calls),
        'router_model_seconds': route_telemetry['gpu_seconds'],
        'model_seconds': sum(call['gpu_seconds'] for call in calls) + route_telemetry['gpu_seconds'],
        'wall_seconds': ground_wall + route_telemetry['wall_seconds'],
        'peak_vram_bytes': max(
            [call['peak_vram_bytes'] for call in calls]
            + [route_telemetry['peak_vram_bytes']]
        ),
    }


def ground_phase(args, spec: BenchmarkSpec, rows: list[dict], run_dir: Path, cache_root: Path) -> None:
    import torch
    from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat

    routes = read_jsonl(run_dir / 'routes.jsonl', ('id', 'budget'))
    predictions_path = run_dir / 'predictions.jsonl'
    complete = read_jsonl(predictions_path, ('id', 'schedule', 'budget', 'prompt_mode'))
    settings = [
        (schedule, budget, prompt_mode)
        for schedule in SCHEDULES for budget in args.budgets for prompt_mode in args.prompt_modes
    ] + [
        (NATIVE_SCHEDULE, args.native_frame_cap, prompt_mode)
        for prompt_mode in args.prompt_modes
    ]
    pending = [
        (row, schedule, budget, prompt_mode)
        for row in rows for schedule, budget, prompt_mode in settings
        if (row['id'], schedule, budget, prompt_mode) not in complete
    ]
    if not pending:
        print(f'Grounding already complete: {len(complete)} records')
        return
    missing = [
        (row['id'], budget) for row in rows for budget in args.budgets
        if (row['id'], budget) not in routes
    ]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} routes; first={missing[0]}')

    print(f'Loading frozen grounder {args.model} for {len(pending)} settings...')
    model = Qwen3VLChat(
        model_path=args.model,
        use_custom_prompt=False,
        use_vllm=False,
        use_audio_in_video=False,
        max_new_tokens=512,
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        fps=None,
        nframe=None,
        min_pixels=32 * 32,
        max_pixels=args.frame_width * args.frame_width,
        total_pixels=args.native_video_tokens * 32 * 32,
        attn_implementation=args.attention,
    )
    for position, (row, schedule, budget, prompt_mode) in enumerate(pending, 1):
        route = routes[(row['id'], budget)] if schedule == 'embedding-window-local' else None
        result = execute_setting(
            model=model, spec=spec, row=row, schedule=schedule,
            prompt_mode=prompt_mode, budget=budget, route=route,
            cache_root=cache_root, frame_width=args.frame_width,
            native_frame_cap=args.native_frame_cap,
        )
        record = {
            **{key: row[key] for key in ('id', 'video', 'duration', 'query', 'targets', 'group')},
            'dataset': spec.name, 'schedule': schedule, 'budget': budget,
            'prompt_mode': prompt_mode,
            'router_recall': route.get('router_recall') if route else None,
            **result,
        }
        append_jsonl(predictions_path, record)
        print(
            f'[ground {position}/{len(pending)}] id={row["id"]} schedule={schedule} '
            f'budget={budget} prompt={prompt_mode}'
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def per_sample_metrics(prediction: list[list[float]], targets: list[list[float]]) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=float) if prediction else np.empty((0, 2), dtype=float)
    gt = np.asarray(targets, dtype=float) if targets else np.empty((0, 2), dtype=float)
    tiou = float(_overlap_ratio(pred, gt))
    return {
        'mIoU': tiou,
        'R@0.1': float(tiou >= 0.1),
        'R@0.2': float(tiou >= 0.2),
        'R@0.3': float(tiou >= 0.3),
        'R@0.5': float(tiou >= 0.5),
        'R@0.7': float(tiou >= 0.7),
        'CardinalityError': float(abs(len(prediction) - len(targets))),
    }


def _cluster_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    clusters: list[str],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError('candidate and reference must be paired one-dimensional arrays')
    unique = sorted(set(clusters))
    members = {cluster: np.flatnonzero(np.asarray(clusters) == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[cluster] for cluster in selected])
        deltas[index] = float(np.mean(candidate[indices] - reference[indices]))
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    p_value = min(1.0, 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return {
        'delta': float(np.mean(candidate - reference)),
        'ci95_low': float(lower),
        'ci95_high': float(upper),
        'p_value': p_value,
    }


def _holm_adjust(rows: list[dict]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]['p_value'])
    adjusted = [0.0] * len(rows)
    running = 0.0
    count = len(rows)
    for rank, (original, row) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * row['p_value']))
        adjusted[original] = running
    for row, value in zip(rows, adjusted):
        row['holm_p_value'] = value
        row['significant_positive'] = value < 0.05 and row['ci95_low'] > 0
        row['significant_negative'] = value < 0.05 and row['ci95_high'] < 0


def _group_metrics(records: list[dict], thresholds: tuple[float, ...]) -> dict[str, dict]:
    output = {}
    for group in sorted({record.get('group', '') for record in records}):
        selected = [record for record in records if record.get('group', '') == group]
        metrics = [
            per_sample_metrics(record['prediction'], record['targets']) for record in selected
        ]
        row = {
            'samples': len(selected),
            'mIoU': 100 * float(np.mean([value['mIoU'] for value in metrics])),
        }
        for threshold in thresholds:
            key = f'R@{threshold:g}'
            row[key] = 100 * float(np.mean([value[key] for value in metrics]))
        output[str(group)] = row
    return output


def _setting_records(predictions: dict, rows: list[dict], schedule: str, budget: int, prompt: str) -> list[dict]:
    return [
        predictions[(row['id'], schedule, budget, prompt)]
        for row in rows
        if (row['id'], schedule, budget, prompt) in predictions
    ]


def evaluate_phase(args, spec: BenchmarkSpec, rows: list[dict], run_dir: Path) -> dict:
    predictions = read_jsonl(
        run_dir / 'predictions.jsonl', ('id', 'schedule', 'budget', 'prompt_mode')
    )
    summaries = []
    setting_keys = [
        (schedule, budget, prompt)
        for schedule in SCHEDULES for budget in args.budgets for prompt in args.prompt_modes
    ] + [
        (NATIVE_SCHEDULE, args.native_frame_cap, prompt) for prompt in args.prompt_modes
    ]
    for schedule, budget, prompt in setting_keys:
        records = _setting_records(predictions, rows, schedule, budget, prompt)
        metric_rows = [per_sample_metrics(record['prediction'], record['targets']) for record in records]
        summary = {
            'schedule': schedule, 'budget': budget, 'prompt_mode': prompt,
            'samples': len(records), 'complete': len(records) == len(rows),
        }
        if metric_rows:
            for key in metric_rows[0]:
                scale = 1.0 if key == 'CardinalityError' else 100.0
                summary[key] = scale * float(np.mean([metric[key] for metric in metric_rows]))
            for key in (
                'grounder_frames', 'embedding_frames', 'total_frames', 'unused_frames',
                'budget_overflow_frames', 'grounder_calls', 'router_calls', 'model_calls',
                'model_seconds', 'wall_seconds',
            ):
                summary[key] = float(np.sum([record[key] for record in records]))
            summary['peak_vram_bytes'] = int(max(record['peak_vram_bytes'] for record in records))
            recalls = [record['router_recall'] for record in records if record['router_recall'] is not None]
            summary['RouterRecall'] = 100 * float(np.mean(recalls)) if recalls else None
            summary[f'by_{spec.grouping_field}'] = _group_metrics(records, spec.thresholds)
            if spec.name == 'vue-tr-v2':
                native_rows = [
                    {
                        'answer': np.asarray(record['prediction'], dtype=float),
                        'gt': np.asarray(record['targets'], dtype=float),
                    }
                    for record in records
                ]
                _, summary['iou_auc'] = _success_overlap(native_rows)
                summary['precision_auc'], summary['recall_auc'] = _compute_precision_recall(native_rows)
                for key in ('iou_auc', 'precision_auc', 'recall_auc'):
                    summary[key] *= 100
        summaries.append(summary)

    comparisons = []
    for budget in args.budgets:
        candidate = _setting_records(
            predictions, rows, 'embedding-window-local', budget, 'controlled'
        )
        reference = _setting_records(
            predictions, rows, 'uniform-one-shot', budget, 'controlled'
        )
        candidate_map = {record['id']: record for record in candidate}
        reference_map = {record['id']: record for record in reference}
        ids = [row['id'] for row in rows if row['id'] in candidate_map and row['id'] in reference_map]
        candidate_values = np.asarray([
            per_sample_metrics(candidate_map[row_id]['prediction'], candidate_map[row_id]['targets'])['mIoU']
            for row_id in ids
        ])
        reference_values = np.asarray([
            per_sample_metrics(reference_map[row_id]['prediction'], reference_map[row_id]['targets'])['mIoU']
            for row_id in ids
        ])
        clusters = [candidate_map[row_id]['video'] for row_id in ids]
        stats = _cluster_bootstrap(
            candidate_values, reference_values, clusters,
            samples=args.bootstrap_samples, seed=args.bootstrap_seed + budget,
        ) if ids else {'delta': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0, 'p_value': 1.0}
        comparisons.append({
            'budget': budget, 'prompt_mode': 'controlled', 'metric': 'mIoU',
            'samples': len(ids), **{key: 100 * value if key != 'p_value' else value for key, value in stats.items()},
        })
    _holm_adjust(comparisons)

    prompt_comparisons = []
    prompt_settings = [
        (schedule, budget)
        for schedule in SCHEDULES for budget in args.budgets
    ] + [(NATIVE_SCHEDULE, args.native_frame_cap)]
    for schedule, budget in prompt_settings:
        controlled = _setting_records(predictions, rows, schedule, budget, 'controlled')
        native = _setting_records(predictions, rows, schedule, budget, 'native-style')
        controlled_map = {record['id']: record for record in controlled}
        native_map = {record['id']: record for record in native}
        ids = [row['id'] for row in rows if row['id'] in controlled_map and row['id'] in native_map]
        controlled_values = np.asarray([
            per_sample_metrics(
                controlled_map[row_id]['prediction'], controlled_map[row_id]['targets']
            )['mIoU']
            for row_id in ids
        ])
        native_values = np.asarray([
            per_sample_metrics(native_map[row_id]['prediction'], native_map[row_id]['targets'])['mIoU']
            for row_id in ids
        ])
        clusters = [controlled_map[row_id]['video'] for row_id in ids]
        stats = _cluster_bootstrap(
            controlled_values, native_values, clusters,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + budget + sum(ord(char) for char in schedule),
        ) if ids else {'delta': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0, 'p_value': 1.0}
        prompt_comparisons.append({
            'schedule': schedule, 'budget': budget,
            'candidate_prompt': 'controlled', 'reference_prompt': 'native-style',
            'metric': 'mIoU', 'samples': len(ids),
            **{key: 100 * value if key != 'p_value' else value for key, value in stats.items()},
        })

    result = {
        'dataset': spec.name,
        'model': args.model,
        'embedding_model': args.embedding_model,
        'primary_comparison': 'embedding-window-local vs uniform-one-shot, controlled prompt',
        'bootstrap': {
            'samples': args.bootstrap_samples, 'seed': args.bootstrap_seed,
            'cluster': 'video', 'multiple_testing': 'Holm across 64/128 budgets',
        },
        'results': summaries,
        'comparisons': comparisons,
        'prompt_comparisons': prompt_comparisons,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    pd.DataFrame(summaries).to_csv(run_dir / 'summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(run_dir / 'comparisons.csv', index=False)
    pd.DataFrame(prompt_comparisons).to_csv(
        run_dir / 'prompt_comparisons.csv', index=False
    )
    write_report(run_dir / 'report.md', result)
    return result


def write_report(path: Path, result: dict) -> None:
    lines = [
        f'# {result["dataset"]}: hierarchical search',
        '',
        '## Co-primary controlled-prompt comparisons',
        '',
        '| Budget | Samples | Δ mIoU | 95% CI | Holm p | Positive |',
        '| ---: | ---: | ---: | ---: | ---: | :---: |',
    ]
    for row in result['comparisons']:
        lines.append(
            f'| {row["budget"]} | {row["samples"]} | {row["delta"]:.2f} | '
            f'[{row["ci95_low"]:.2f}, {row["ci95_high"]:.2f}] | '
            f'{row["holm_p_value"]:.4f} | {"yes" if row["significant_positive"] else "no"} |'
        )
    lines.extend([
        '',
        '## All settings',
        '',
        '| Schedule | Budget | Prompt | mIoU | R@0.3 | R@0.5 | R@0.7 | Frames | Calls | Model s | Wall s |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in result['results']:
        lines.append(
            f'| {row["schedule"]} | {row["budget"]} | {row["prompt_mode"]} | '
            f'{row.get("mIoU", 0):.2f} | {row.get("R@0.3", 0):.2f} | '
            f'{row.get("R@0.5", 0):.2f} | {row.get("R@0.7", 0):.2f} | '
            f'{row.get("total_frames", 0):.0f} | {row.get("model_calls", 0):.0f} | '
            f'{row.get("model_seconds", 0):.1f} | {row.get("wall_seconds", 0):.1f} |'
        )
    lines.extend([
        '',
        '## Controlled versus native-style prompt',
        '',
        '| Schedule | Budget | Δ mIoU | 95% CI |',
        '| --- | ---: | ---: | ---: |',
    ])
    for row in result['prompt_comparisons']:
        lines.append(
            f'| {row["schedule"]} | {row["budget"]} | {row["delta"]:.2f} | '
            f'[{row["ci95_low"]:.2f}, {row["ci95_high"]:.2f}] |'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def validate_inputs(args, spec: BenchmarkSpec, rows: list[dict], metadata: dict) -> None:
    if not rows:
        raise ValueError(f'No usable rows found for {spec.name}')
    missing = [row['video_path'] for row in rows if not Path(row['video_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} videos; first={missing[0]}')
    if spec.name == 'vue-tr-v2' and metadata['coverage'] < args.minimum_vue_coverage:
        raise RuntimeError(
            f'VUE-TR-V2 coverage {metadata["coverage"]:.1%} is below '
            f'{args.minimum_vue_coverage:.1%}'
        )
    for row in rows:
        if row['duration'] <= 0 or not row['query'] or not row['targets']:
            raise ValueError(f'Invalid normalized row: {row["id"]}')
    print(
        f'Validation passed: dataset={spec.name} rows={len(rows)} '
        f'coverage={metadata["coverage"]:.1%}'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(BENCHMARKS), required=True)
    parser.add_argument('--phase', choices=('validate', 'route', 'ground', 'evaluate', 'all'), default='all')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument('--run-name', default='smoke')
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--model', default='MCG-NJU/TimeLens2-4B')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-VL-Embedding-2B')
    parser.add_argument('--budgets', default='64,128')
    parser.add_argument('--prompt-modes', default=','.join(PROMPT_MODES))
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--frame-width', type=int, default=336)
    parser.add_argument('--embedding-max-pixels', type=int, default=336 * 336)
    parser.add_argument('--native-frame-cap', type=int, default=512)
    parser.add_argument('--native-video-tokens', type=int, default=8192)
    parser.add_argument('--attention', default='sdpa')
    parser.add_argument('--minimum-vue-coverage', type=float, default=0.90)
    parser.add_argument('--bootstrap-samples', type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--keep-frame-cache', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = BENCHMARKS[args.dataset]
    args.budgets = parse_csv(args.budgets, int)
    args.prompt_modes = parse_csv(args.prompt_modes)
    if any(budget < 4 or budget % 2 for budget in args.budgets):
        raise ValueError('Budgets must be even and at least four')
    if set(args.prompt_modes) - set(PROMPT_MODES):
        raise ValueError(f'Unknown prompt modes: {set(args.prompt_modes) - set(PROMPT_MODES)}')
    if args.native_frame_cap < 2 or args.native_frame_cap % 2:
        raise ValueError('native-frame-cap must be an even integer of at least two')
    if not 0 <= args.minimum_vue_coverage <= 1:
        raise ValueError('minimum-vue-coverage must be between zero and one')

    args.data_root = args.data_root or Path(os.environ.get(spec.root_env, spec.default_root))
    args.cache_root = args.cache_root or Path(os.environ.get(
        'TIMELENS2_SEARCH_CACHE', '/tmp/timelens2-vtg-search'
    ))
    run_dir = args.output_root / args.run_name / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root / args.run_name / spec.name
    rows, metadata = load_benchmark(spec, args.data_root, args.max_samples)
    annotation = Path(metadata['annotation'])
    metadata['annotation_sha256'] = sha256_file(annotation)
    manifest = {
        'dataset': spec.name,
        'metadata': metadata,
        'rows': [
            {
                key: row[key] for key in
                ('id', 'video', 'video_path', 'duration', 'query', 'targets', 'group')
            }
            for row in rows
        ],
    }
    manifest_path = run_dir / 'dataset_manifest.json'
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if manifest_path.is_file() and manifest_path.read_text(encoding='utf-8') != manifest_text:
        raise RuntimeError('Dataset manifest changed; use a new --run-name')
    manifest_path.write_text(manifest_text, encoding='utf-8')

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ('phase', 'keep_frame_cache')
    }
    config['annotation_sha256'] = metadata['annotation_sha256']
    config_path = run_dir / 'config.json'
    config_text = json.dumps(config, indent=2, sort_keys=True) + '\n'
    if config_path.is_file() and config_path.read_text(encoding='utf-8') != config_text:
        raise RuntimeError('Run configuration changed; use a new --run-name')
    config_path.write_text(config_text, encoding='utf-8')
    append_jsonl(run_dir / 'invocations.jsonl', {
        'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'phase': args.phase,
        'source_revision': os.environ.get('TIMELENS2_SOURCE_REVISION', 'unknown'),
    })

    if args.phase in ('validate', 'all'):
        validate_inputs(args, spec, rows, metadata)
    if args.phase in ('route', 'all'):
        route_phase(args, spec, rows, run_dir, cache_root)
    if args.phase in ('ground', 'all'):
        ground_phase(args, spec, rows, run_dir, cache_root)
    if args.phase in ('evaluate', 'all'):
        evaluate_phase(args, spec, rows, run_dir)
    if args.phase == 'all' and not args.keep_frame_cache and cache_root.is_dir():
        shutil.rmtree(cache_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
