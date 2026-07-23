#!/usr/bin/env python3
"""Run the hierarchical TimeLens2 OMTG experiment on a single GPU.

The command is deliberately phase-oriented and append-only.  A disconnected
Colab client can restart the same command and continue from routes.jsonl and
predictions.jsonl without repeating completed examples.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.dataset.omtgbench import compute_one_to_many_metrics, parse_time_intervals
from vlmeval.omtg_search import (
    CallTimer,
    Window,
    append_jsonl,
    consolidate_intervals,
    content_aware_windows,
    distribute_frames,
    extract_frames,
    oracle_window_indices,
    probe_video,
    read_jsonl,
    retained_window_count,
    router_recall,
    uniform_timestamps,
    uniform_window_indices,
)


ALL_SCHEDULES = (
    'uniform-one-shot',
    'full-video-multipass',
    'uniform-window-local',
    'embedding-window-local',
)


def parse_csv(value: str, converter=str):
    return [converter(item.strip()) for item in value.split(',') if item.strip()]


def query_text(question: str) -> str:
    match = re.search(r"given textual query ['\"](.+?)['\"]", question, re.IGNORECASE)
    return match.group(1) if match else question


def load_rows(root: Path, max_samples: int) -> list[dict]:
    data_file = root / 'OMTGBench.tsv'
    if not data_file.is_file():
        raise FileNotFoundError(f'Missing {data_file}; run scripts/download_omtg_bench.sh first')
    data = pd.read_csv(data_file, sep='\t')
    if max_samples > 0:
        data = data.iloc[:max_samples]
    return data.to_dict(orient='records')


def frame_dir(cache_root: Path, video: str, label: str) -> Path:
    safe_video = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(video).stem)
    safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
    return cache_root / safe_video / safe_label


def load_windows_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def save_windows_cache(path: Path, value: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def route_phase(args, rows: list[dict], run_dir: Path, cache_root: Path) -> None:
    import torch
    from vlmeval.vlm.qwen3_vl_embedding import Qwen3VLEmbedder

    route_path = run_dir / 'routes.jsonl'
    completed = read_jsonl(route_path, ('id',))
    pending = [row for row in rows if (int(row['id']),) not in completed]
    if not pending:
        print(f'Routing already complete: {len(completed)} records')
        return

    print(f'Loading router {args.embedding_model} for {len(pending)} records...')
    embedder = Qwen3VLEmbedder(
        args.embedding_model,
        max_pixels=args.embedding_max_pixels,
        total_pixels=4 * args.embedding_max_pixels,
        attn_implementation=args.attention,
    )
    windows_cache_path = run_dir / 'video_windows.json'
    windows_cache = load_windows_cache(windows_cache_path)

    for position, row in enumerate(pending, 1):
        route_started = time.perf_counter()
        video_path = args.data_root / 'videos' / row['video']
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        windowing_started = time.perf_counter()
        if row['video'] not in windows_cache:
            windows, source = content_aware_windows(str(video_path))
            windows_cache[row['video']] = {
                'source': source,
                'duration': probe_video(str(video_path))['duration'],
                'windows': [window.to_dict() for window in windows],
            }
            save_windows_cache(windows_cache_path, windows_cache)
        windowing_wall_seconds = time.perf_counter() - windowing_started
        metadata = windows_cache[row['video']]
        windows = [Window(**item) for item in metadata['windows']]
        sparse_frame_count = 0
        embedding_timers = []
        with CallTimer() as query_timer:
            query_embedding = embedder.process([{
                'text': query_text(str(row['question'])),
                'instruction': 'Represent this text for retrieving matching temporal video windows.',
            }])[0]
        embedding_timers.append(query_timer.to_dict())
        scores = []
        for index, window in enumerate(windows):
            timestamps = uniform_timestamps(window.start, window.end, 4)
            frames = extract_frames(
                str(video_path), timestamps,
                frame_dir(cache_root, row['video'], f'router_{index}'),
                max_width=args.frame_width,
            )
            with CallTimer() as video_timer:
                video_embedding = embedder.process([{
                    'video': frames,
                    'sample_fps': 4 / max(window.duration, 1e-3),
                    'instruction': 'Represent this video window for retrieval by a textual event description.',
                }])[0]
            embedding_timers.append(video_timer.to_dict())
            scores.append(float(torch.dot(query_embedding.float(), video_embedding.float()).item()))
            sparse_frame_count += len(frames)
        telemetry = {
            'wall_seconds': round(time.perf_counter() - route_started, 6),
            'gpu_seconds': round(sum(item['gpu_seconds'] for item in embedding_timers), 6),
            'peak_vram_bytes': max(item['peak_vram_bytes'] for item in embedding_timers),
            'embedding_calls': len(embedding_timers),
        }
        count = retained_window_count(len(windows))
        selected = sorted(range(len(windows)), key=lambda index: (-scores[index], index))[:count]
        targets = parse_time_intervals(str(row['answer']))
        oracle = oracle_window_indices(windows, targets, count)
        record = {
            'id': int(row['id']),
            'video': row['video'],
            'duration': metadata['duration'],
            'window_source': metadata['source'],
            'windows': [window.to_dict() for window in windows],
            'scores': scores,
            'selected': selected,
            'k': count,
            'embedding_frames': sparse_frame_count,
            'router_recall': router_recall(windows, selected, targets),
            'oracle_selected': oracle,
            'oracle_router_recall': router_recall(windows, oracle, targets),
            'windowing_wall_seconds': round(windowing_wall_seconds, 6),
            'telemetry': telemetry,
        }
        append_jsonl(route_path, record)
        print(f'[route {position}/{len(pending)}] id={row["id"]} windows={len(windows)} k={count}')

    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def grounding_prompt(query: str, duration: float, local: bool) -> str:
    coordinate = 'relative to the start of this clip' if local else 'relative to the full video'
    return (
        f'Find every disjoint time interval where this event occurs: {query!r}. '
        f'Timestamps must be in seconds {coordinate}, between 0 and {duration:.3f}. '
        'Return only a JSON array of [start, end] pairs, ordered by start time. '
        'Return [] when the event does not occur. Do not omit repeated occurrences.'
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
        response = model.generate(message, dataset='OMTGBench')
    return response, parse_time_intervals(response), timer.to_dict()


def execute_schedule(
    *,
    model,
    schedule: str,
    budget: int,
    row: dict,
    route: dict,
    video_path: Path,
    cache_root: Path,
    frame_width: int,
) -> dict:
    windows = [Window(**value) for value in route['windows']]
    duration = float(route['duration'])
    query = query_text(str(row['question']))
    calls: list[dict] = []
    global_intervals: list[list[float]] = []
    frames_used = 0
    ground_started = time.perf_counter()

    def call_window(window: Window, count: int, label: str, phase: float = 0.5, local: bool = True):
        nonlocal frames_used
        timestamps = uniform_timestamps(window.start, window.end, count, phase=phase)
        frames = extract_frames(
            str(video_path), timestamps, frame_dir(cache_root, row['video'], label), max_width=frame_width
        )
        response, intervals, telemetry = timed_ground_call(
            model, frames, grounding_prompt(query, window.duration if local else duration, local), window.duration
        )
        if local:
            intervals = [
                [max(0.0, start) + window.start, min(window.duration, end) + window.start]
                for start, end in intervals
                if min(window.duration, end) > max(0.0, start)
            ]
        global_intervals.extend(intervals)
        frames_used += len(frames)
        calls.append({
            'window': window.to_dict(),
            'frame_count': len(frames),
            'response': response,
            **telemetry,
        })

    if schedule == 'uniform-one-shot':
        call_window(Window(0.0, duration), budget, f'{schedule}_{budget}', local=False)
    elif schedule == 'full-video-multipass':
        allocations = distribute_frames(budget, 2)
        call_window(Window(0.0, duration), allocations[0], f'{schedule}_{budget}_0', phase=0.25, local=False)
        call_window(Window(0.0, duration), allocations[1], f'{schedule}_{budget}_1', phase=0.75, local=False)
    elif schedule in ('uniform-window-local', 'embedding-window-local'):
        selected = (
            uniform_window_indices(len(windows), int(route['k']))
            if schedule == 'uniform-window-local'
            else [int(index) for index in route['selected']]
        )
        local_budget = budget
        if schedule == 'embedding-window-local':
            # Four router frames per content window count against the nominal
            # decoded-frame budget. Two local frames/window is the feasibility
            # floor; any overflow is explicit in the output record.
            local_budget = max(2 * len(selected), budget - int(route['embedding_frames']))
        allocations = distribute_frames(local_budget, len(selected))
        for allocation, index in zip(allocations, selected):
            call_window(windows[index], allocation, f'{schedule}_{budget}_{index}', local=True)
    else:
        raise ValueError(f'Unknown schedule: {schedule}')

    prediction = consolidate_intervals(global_intervals, duration=duration)
    ground_gpu = sum(call['gpu_seconds'] for call in calls)
    ground_wall = time.perf_counter() - ground_started
    route_gpu = route['telemetry']['gpu_seconds'] if schedule == 'embedding-window-local' else 0.0
    if schedule == 'embedding-window-local':
        route_wall = route['telemetry']['wall_seconds']
    elif schedule == 'uniform-window-local':
        route_wall = route.get('windowing_wall_seconds', 0.0)
    else:
        route_wall = 0.0
    embedding_frames = route['embedding_frames'] if schedule == 'embedding-window-local' else 0
    return {
        'prediction': prediction,
        'raw_intervals': global_intervals,
        'calls': calls,
        'model_calls': len(calls),
        'grounder_frames': frames_used,
        'embedding_frames': embedding_frames,
        'total_frames': frames_used + embedding_frames,
        'budget_overflow_frames': max(0, frames_used + embedding_frames - budget),
        'grounder_gpu_seconds': ground_gpu,
        'grounder_wall_seconds': ground_wall,
        'router_gpu_seconds': route_gpu,
        'gpu_seconds': ground_gpu + route_gpu,
        'wall_seconds': ground_wall + route_wall,
        'peak_vram_bytes': max([call['peak_vram_bytes'] for call in calls] + [
            route['telemetry']['peak_vram_bytes'] if schedule == 'embedding-window-local' else 0
        ]),
    }


def ground_phase(args, rows: list[dict], run_dir: Path, cache_root: Path) -> None:
    import torch
    from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat

    routes = read_jsonl(run_dir / 'routes.jsonl', ('id',))
    missing = [int(row['id']) for row in rows if (int(row['id']),) not in routes]
    if missing:
        raise RuntimeError(f'Missing routes for {len(missing)} rows; first id={missing[0]}')
    prediction_path = run_dir / 'predictions.jsonl'
    completed = read_jsonl(prediction_path, ('id', 'schedule', 'budget'))
    expected = len(rows) * len(args.schedules) * len(args.budgets)
    if len(completed) >= expected:
        print(f'Grounding already complete: {len(completed)} records')
        return

    print(f'Loading frozen grounder {args.model}...')
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
        total_pixels=8192 * 32 * 32,
        attn_implementation=args.attention,
    )
    position = 0
    for row in rows:
        row_id = int(row['id'])
        route = routes[(row_id,)]
        video_path = args.data_root / 'videos' / row['video']
        for budget in args.budgets:
            for schedule in args.schedules:
                key = (row_id, schedule, budget)
                if key in completed:
                    continue
                position += 1
                result = execute_schedule(
                    model=model,
                    schedule=schedule,
                    budget=budget,
                    row=row,
                    route=route,
                    video_path=video_path,
                    cache_root=cache_root,
                    frame_width=args.frame_width,
                )
                record = {
                    'id': row_id,
                    'video': row['video'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'schedule': schedule,
                    'budget': budget,
                    'router_recall': route['router_recall'] if schedule == 'embedding-window-local' else None,
                    **result,
                }
                append_jsonl(prediction_path, record)
                print(f'[ground {position}] id={row_id} schedule={schedule} budget={budget} '
                      f'predicted={len(result["prediction"])}')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_phase(args, rows: list[dict], run_dir: Path) -> dict:
    predictions = read_jsonl(run_dir / 'predictions.jsonl', ('id', 'schedule', 'budget'))
    routes = read_jsonl(run_dir / 'routes.jsonl', ('id',))
    summaries = []
    for budget in args.budgets:
        for schedule in args.schedules:
            records = [predictions.get((int(row['id']), schedule, budget)) for row in rows]
            records = [record for record in records if record is not None]
            metric_rows = [
                compute_one_to_many_metrics(record['prediction'], parse_time_intervals(str(record['answer'])))
                for record in records
            ]
            summary = {
                'schedule': schedule,
                'budget': budget,
                'samples': len(records),
                'complete': len(records) == len(rows),
            }
            if metric_rows:
                for key in metric_rows[0]:
                    scale = 1.0 if key == 'CardinalityError' else 100.0
                    summary[key] = float(np.mean([metric[key] for metric in metric_rows])) * scale
                for key in ('grounder_frames', 'embedding_frames', 'total_frames', 'budget_overflow_frames',
                            'model_calls', 'gpu_seconds', 'wall_seconds'):
                    summary[key] = float(np.sum([record[key] for record in records]))
                summary['peak_vram_bytes'] = float(np.max([record['peak_vram_bytes'] for record in records]))
                if schedule == 'embedding-window-local':
                    summary['RouterRecall'] = 100 * float(np.mean([
                        routes[(int(record['id']),)]['router_recall'] for record in records
                    ]))
                    summary['OracleRouterRecall'] = 100 * float(np.mean([
                        routes[(int(record['id']),)]['oracle_router_recall'] for record in records
                    ]))
            summaries.append(summary)

    for budget in args.budgets:
        reference = next((item for item in summaries
                          if item['budget'] == budget and item['schedule'] == 'uniform-one-shot'), None)
        if not reference or not reference.get('gpu_seconds'):
            continue
        for item in summaries:
            if item['budget'] != budget or not item.get('gpu_seconds'):
                continue
            relative = item['gpu_seconds'] / reference['gpu_seconds']
            item['gpu_time_ratio_vs_uniform'] = relative
            item['compute_matched_within_5pct'] = abs(relative - 1.0) <= 0.05

    result = {
        'dataset': 'OMTGBench',
        'model': args.model,
        'embedding_model': args.embedding_model,
        'metric_units': 'percent except CardinalityError and efficiency counters',
        'compute_policy': 'matched only when aggregate GPU time differs by at most 5%; otherwise compare Pareto points',
        'results': summaries,
    }
    (run_dir / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    pd.DataFrame(summaries).to_csv(run_dir / 'summary.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('route', 'ground', 'evaluate', 'all'), default='all')
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/omtg_search'))
    parser.add_argument('--run-name', default='smoke')
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--model', default='MCG-NJU/TimeLens2-4B')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-VL-Embedding-2B')
    parser.add_argument('--budgets', default='32,64')
    parser.add_argument('--schedules', default=','.join(ALL_SCHEDULES))
    parser.add_argument('--max-samples', type=int, default=25,
                        help='First N deterministic rows; use 0 for the full 320-row benchmark.')
    parser.add_argument('--frame-width', type=int, default=336)
    parser.add_argument('--embedding-max-pixels', type=int, default=336 * 336)
    parser.add_argument('--attention', default='sdpa')
    parser.add_argument('--keep-frame-cache', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import os

    base_data = Path(os.environ.get('TIMELENS2_DATA_ROOT', '/content/timelens2-data'))
    args.data_root = args.data_root or Path(os.environ.get('OMTG_BENCH_ROOT', base_data / 'OMTGBench'))
    args.cache_root = args.cache_root or Path(os.environ.get(
        'TIMELENS2_SEARCH_CACHE', '/content/timelens2-cache/omtg-search'
    ))
    args.budgets = parse_csv(args.budgets, int)
    args.schedules = parse_csv(args.schedules)
    unknown = sorted(set(args.schedules) - set(ALL_SCHEDULES))
    if unknown:
        raise ValueError(f'Unknown schedules: {unknown}')
    if any(budget < 4 or budget % 2 for budget in args.budgets):
        raise ValueError('Budgets must be even and at least four frames')

    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root / args.run_name
    rows = load_rows(args.data_root, args.max_samples)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ('phase', 'keep_frame_cache')
    }
    config_path = run_dir / 'config.json'
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous != config:
            changed = sorted(key for key in set(previous) | set(config) if previous.get(key) != config.get(key))
            raise RuntimeError(
                f'Run {args.run_name!r} already exists with incompatible settings {changed}; use a new --run-name.'
            )
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    append_jsonl(run_dir / 'invocations.jsonl', {
        'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'phase': args.phase,
        'colab_job_id': os.environ.get('TIMELENS2_COLAB_JOB_ID'),
        'source_revision': os.environ.get('TIMELENS2_SOURCE_REVISION', 'unknown'),
        'source_archive_sha256': os.environ.get('TIMELENS2_SOURCE_ARCHIVE_SHA256', 'unknown'),
    })

    if args.phase in ('route', 'all'):
        route_phase(args, rows, run_dir, cache_root)
    if args.phase in ('ground', 'all'):
        ground_phase(args, rows, run_dir, cache_root)
    if args.phase in ('evaluate', 'all'):
        evaluate_phase(args, rows, run_dir)
    if args.phase == 'all' and not args.keep_frame_cache and cache_root.is_dir():
        shutil.rmtree(cache_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
