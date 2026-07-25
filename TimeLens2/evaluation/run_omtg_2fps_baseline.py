#!/usr/bin/env python3
"""Run resumable paper-style 2 FPS TimeLens2 inference on OMTG Bench."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.dataset.omtgbench import compute_one_to_many_metrics, parse_time_intervals
from vlmeval.omtg_search import (
    CallTimer,
    append_jsonl,
    estimated_sampled_frames,
    grounding_prompt,
    probe_video,
    query_text,
    read_jsonl,
)

PROMPT_MODES = ('official', 'controlled')
PAPER_FPS = 2.0
PAPER_MIN_PIXELS = 2048
PAPER_TOTAL_PIXELS = 8_388_608
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / 'results'


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def load_rows(root: Path, max_samples: int) -> list[dict]:
    data_file = root / 'OMTGBench.tsv'
    if not data_file.is_file():
        raise FileNotFoundError(f'Missing {data_file}; run scripts/download_omtg_bench.sh first')
    data = pd.read_csv(data_file, sep='\t')
    if max_samples > 0:
        data = data.iloc[:max_samples]
    return data.to_dict(orient='records')


def prompt_for_row(row: dict, mode: str, duration: float) -> str:
    if mode == 'official':
        return str(row['question'])
    if mode == 'controlled':
        return grounding_prompt(query_text(str(row['question'])), duration, local=False)
    raise ValueError(f'Unknown prompt mode: {mode}')


def infer_one(model, row: dict, video_path: Path, mode: str) -> dict:
    metadata = probe_video(str(video_path))
    duration = float(metadata['duration'])
    prompt = prompt_for_row(row, mode, duration)
    message = [
        {'type': 'video', 'value': str(video_path), 'fps': PAPER_FPS},
        {'type': 'text', 'value': prompt},
    ]
    with CallTimer() as timer:
        response = model.generate(message, dataset='OMTGBench')
    return {
        'prompt_mode': mode,
        'prompt': prompt,
        'response': response,
        'prediction': parse_time_intervals(response),
        'duration': duration,
        'source_fps': float(metadata['fps']),
        'source_frames': int(metadata['frames']),
        'requested_frames_estimate': estimated_sampled_frames(duration, PAPER_FPS),
        'model_calls': 1,
        **timer.to_dict(),
    }


def load_model(args):
    from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat

    return Qwen3VLChat(
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
        fps=PAPER_FPS,
        nframe=None,
        min_pixels=PAPER_MIN_PIXELS,
        max_pixels=None,
        total_pixels=PAPER_TOTAL_PIXELS,
        attn_implementation=args.attention,
    )


def inference_phase(args, rows: list[dict], run_dir: Path) -> None:
    predictions_path = run_dir / 'predictions.jsonl'
    completed = read_jsonl(predictions_path, ('id', 'prompt_mode'))
    expected = len(rows) * len(args.prompt_modes)
    if len(completed) >= expected:
        print(f'Inference already complete: {len(completed)} records')
        return

    print(f'Loading frozen grounder {args.model}...')
    model = load_model(args)
    position = len(completed)
    for row in rows:
        row_id = int(row['id'])
        video_path = args.data_root / 'videos' / str(row['video'])
        if not video_path.is_file():
            raise FileNotFoundError(f'Missing benchmark video: {video_path}')
        for mode in args.prompt_modes:
            if (row_id, mode) in completed:
                continue
            result = infer_one(model, row, video_path, mode)
            append_jsonl(predictions_path, {
                'id': row_id,
                'video': row['video'],
                'question': row['question'],
                'answer': row['answer'],
                **result,
            })
            position += 1
            print(f'[baseline {position}/{expected}] id={row_id} prompt={mode} '
                  f'predicted={len(result["prediction"])}')

    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate_phase(args, rows: list[dict], run_dir: Path) -> dict:
    predictions = read_jsonl(run_dir / 'predictions.jsonl', ('id', 'prompt_mode'))
    summaries = []
    for mode in args.prompt_modes:
        records = [predictions.get((int(row['id']), mode)) for row in rows]
        records = [record for record in records if record is not None]
        metric_rows = [
            compute_one_to_many_metrics(
                # The raw response is authoritative so improved parsers can
                # rescore completed inference without rerunning the model.
                parse_time_intervals(record['response'])
                if 'response' in record else record['prediction'],
                parse_time_intervals(str(record['answer'])),
            )
            for record in records
        ]
        summary = {
            'schedule': f'paper-2fps-{mode}',
            'prompt_mode': mode,
            'fps': PAPER_FPS,
            'min_pixels': PAPER_MIN_PIXELS,
            'total_pixels': PAPER_TOTAL_PIXELS,
            'samples': len(records),
            'complete': len(records) == len(rows),
        }
        if metric_rows:
            for key in metric_rows[0]:
                scale = 1.0 if key == 'CardinalityError' else 100.0
                summary[key] = float(np.mean([metric[key] for metric in metric_rows])) * scale
            for key in ('requested_frames_estimate', 'model_calls', 'gpu_seconds', 'wall_seconds'):
                summary[key] = float(np.sum([record[key] for record in records]))
            summary['peak_vram_bytes'] = float(
                np.max([record['peak_vram_bytes'] for record in records])
            )
        summaries.append(summary)

    result = {
        'dataset': 'OMTGBench',
        'model': args.model,
        'backend': 'transformers',
        'visual_policy': {
            'fps': PAPER_FPS,
            'min_pixels': PAPER_MIN_PIXELS,
            'total_pixels': PAPER_TOTAL_PIXELS,
            'max_pixels': None,
            'audio': False,
        },
        'metric_units': 'percent except CardinalityError and efficiency counters',
        'frame_count_note': 'requested_frames_estimate is duration-derived and factor-aligned, not decoder telemetry',
        'results': summaries,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    pd.DataFrame(summaries).to_csv(run_dir / 'summary.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('infer', 'evaluate', 'all'), default='all')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=DEFAULT_RESULTS_ROOT / 'omtg_2fps',
    )
    parser.add_argument('--run-name', default='timelens2-4b-paper-2fps')
    parser.add_argument('--model', default='MCG-NJU/TimeLens2-4B')
    parser.add_argument('--prompt-modes', default=','.join(PROMPT_MODES))
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--attention', default='sdpa')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.prompt_modes = parse_csv(args.prompt_modes)
    unknown = sorted(set(args.prompt_modes) - set(PROMPT_MODES))
    if unknown:
        raise ValueError(f'Unknown prompt modes: {unknown}')
    if not args.prompt_modes:
        raise ValueError('At least one prompt mode is required')

    rows = load_rows(args.data_root, args.max_samples)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'attention': args.attention,
        'backend': 'transformers',
        'data_root': str(args.data_root),
        'max_samples': args.max_samples,
        'model': args.model,
        'output_root': str(args.output_root),
        'prompt_modes': args.prompt_modes,
        'run_name': args.run_name,
        'visual_policy': {
            'fps': PAPER_FPS,
            'min_pixels': PAPER_MIN_PIXELS,
            'total_pixels': PAPER_TOTAL_PIXELS,
            'max_pixels': None,
            'audio': False,
        },
    }
    config_path = run_dir / 'config.json'
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous != config:
            changed = sorted(
                key for key in set(previous) | set(config)
                if previous.get(key) != config.get(key)
            )
            raise RuntimeError(
                f'Run {args.run_name!r} already exists with incompatible settings {changed}; '
                'use a new --run-name.'
            )
    else:
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
    append_jsonl(run_dir / 'invocations.jsonl', {
        'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'phase': args.phase,
    })

    if args.phase in ('infer', 'all'):
        inference_phase(args, rows, run_dir)
    if args.phase in ('evaluate', 'all'):
        evaluate_phase(args, rows, run_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
