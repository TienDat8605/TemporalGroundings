#!/usr/bin/env python3
"""Run training-free hierarchical temporal and spatial VTG inference."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_vtg_search import BENCHMARKS, load_benchmark, per_sample_metrics
from vlmeval.benchmark_adapters import ADAPTERS, load_adapter, validate_canonical_rows
from vlmeval.boundary_refinement import BoundaryRefinementConfig, refine_interval
from vlmeval.dataset.omtgbench import parse_time_intervals
from vlmeval.hybrid_temporal import (
    Candidate,
    CoarseIndex,
    index_cache_key,
    load_index,
    multiscale_candidates,
    save_index,
    score_candidates,
    select_candidates,
    sha256_file,
)
from vlmeval.omtg_search import (
    Window,
    append_jsonl,
    consolidate_intervals,
    content_aware_windows,
    distribute_frames,
    extract_frames,
    read_jsonl,
    uniform_timestamps,
)
from vlmeval.proposal_scorer import ProposalScorerConfig, evidence_curve, propose_intervals
from vlmeval.spatial_pruning import (
    SpatialPruningConfig,
    SpatialPruningResult,
    motion_importance,
    prune_spatial_tokens,
    query_relevance,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT.parent / 'results' / 'hybrid_vtg'
DATASETS = tuple(sorted((*BENCHMARKS, *ADAPTERS)))
INDEX_INSTRUCTION = 'Represent this video frame for reusable temporal event retrieval.'
QUERY_INSTRUCTION = 'Represent this text for retrieving matching temporal video evidence.'


def parse_csv(value: str, converter=str) -> list:
    return [converter(item.strip()) for item in value.split(',') if item.strip()]


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'not JSON serializable: {type(value)!r}')


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def load_rows(args) -> tuple[list[dict], dict, str]:
    if args.dataset in ADAPTERS:
        split = args.split
        if split == 'auto':
            split = 'val_1' if args.dataset == 'activitynet-captions' else 'test'
        args.split = split
        if args.data_root is not None:
            root = args.data_root
        else:
            root_value = os.environ.get(ADAPTERS[args.dataset].root_env)
            if not root_value:
                raise ValueError(f'--data-root or {ADAPTERS[args.dataset].root_env} is required')
            root = Path(root_value)
        if not root:
            raise ValueError(f'--data-root or {ADAPTERS[args.dataset].root_env} is required')
        rows, metadata = load_adapter(
            args.dataset,
            root,
            split=split,
            maximum=args.max_samples,
        )
        validate_canonical_rows(rows)
        return rows, metadata, ADAPTERS[args.dataset].cardinality
    spec = BENCHMARKS[args.dataset]
    root = args.data_root or Path(os.environ.get(spec.root_env, spec.default_root))
    rows, metadata = load_benchmark(spec, root, args.max_samples)
    for row in rows:
        row['cardinality'] = spec.cardinality
        row['modalities'] = ['video']
        row['native_protocol'] = 'continuous-seconds'
    validate_canonical_rows(rows)
    return rows, metadata, spec.cardinality


def validate_inputs(rows: list[dict]) -> None:
    missing = [row['video_path'] for row in rows if not Path(row['video_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'missing {len(missing)} videos; first={missing[0]}')


def _index_manifest(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def index_phase(args, rows: list[dict], run_dir: Path, cache_root: Path) -> None:
    from vlmeval.vlm.qwen3_vl_embedding import Qwen3VLEmbedder

    manifest_path = run_dir / 'coarse_index.json'
    manifest = _index_manifest(manifest_path)
    videos = {}
    for row in rows:
        videos.setdefault(row['video'], row)
    pending = [row for video, row in videos.items() if video not in manifest]
    if not pending:
        return
    embedder = Qwen3VLEmbedder(
        args.embedding_model,
        max_pixels=args.embedding_max_pixels,
        total_pixels=args.embedding_max_pixels,
        attn_implementation=args.attention,
    )
    for row in pending:
        video_path = Path(row['video_path'])
        video_hash = sha256_file(video_path)
        key = index_cache_key(
            video_hash,
            args.embedding_model,
            args.coarse_fps,
            args.frame_width,
            INDEX_INSTRUCTION,
        )
        index_path = cache_root / 'indices' / f'{key}.npz'
        count = max(1, int(math.ceil(float(row['duration']) * args.coarse_fps)))
        timestamps = uniform_timestamps(0.0, float(row['duration']), count)
        frames = extract_frames(
            str(video_path),
            timestamps,
            cache_root / 'frames' / key / 'coarse',
            max_width=args.frame_width,
        )
        features = embedder.process_frames(
            frames,
            instruction=INDEX_INSTRUCTION,
            batch_size=args.embedding_batch_size,
        ).float().cpu().numpy()
        save_index(index_path, CoarseIndex(
            video_path=str(video_path),
            video_hash=video_hash,
            checkpoint=args.embedding_model,
            fps=args.coarse_fps,
            timestamps=np.asarray(timestamps, dtype=np.float64),
            features=features,
        ))
        manifest[row['video']] = {
            'path': str(index_path),
            'video_path': str(video_path),
            'video_hash': video_hash,
            'frames': len(timestamps),
        }
        atomic_json(manifest_path, manifest)
    del embedder
    gc.collect()


def route_phase(args, rows: list[dict], run_dir: Path) -> None:
    from vlmeval.vlm.qwen3_vl_embedding import Qwen3VLEmbedder

    indices = _index_manifest(run_dir / 'coarse_index.json')
    routes_path = run_dir / 'routes.jsonl'
    complete = read_jsonl(routes_path, ('id',))
    pending = [row for row in rows if (row['id'],) not in complete]
    if not pending:
        return
    embedder = Qwen3VLEmbedder(
        args.embedding_model,
        max_pixels=args.embedding_max_pixels,
        total_pixels=args.embedding_max_pixels,
        attn_implementation=args.attention,
    )
    for row in pending:
        index = load_index(Path(indices[row['video']]['path']))
        query = embedder.process([{
            'text': row['query'],
            'instruction': QUERY_INSTRUCTION,
        }])[0].float().cpu().numpy()
        content, content_source = content_aware_windows(str(row['video_path']))
        if args.temporal_policy == 'uniform-full':
            candidates = [Candidate(
                index=0,
                start=0.0,
                end=float(row['duration']),
                scale=float(row['duration']),
                source='uniform-full',
            )]
        else:
            candidates = multiscale_candidates(
                float(row['duration']),
                scales=args.window_scales if args.temporal_policy == 'hybrid' else (),
                stride_ratio=args.window_stride_ratio,
                content_windows=content,
            )
        candidates = score_candidates(candidates, index, query, args.mean_score_weight)
        if args.temporal_policy == 'uniform-full':
            route_components = [{
                'start': 0.0,
                'end': float(row['duration']),
                'source_candidates': (0,),
                'score': float(candidates[0].score),
            }]
            selected = [0]
            confidence_margin = 0.0
            low_confidence_fallback = False
            retained_union_seconds = float(row['duration'])
        else:
            route = select_candidates(
                candidates,
                float(row['duration']),
                union_budget_seconds=args.temporal_union_budget,
                maximum_candidates=args.maximum_candidates,
                nms_iou=args.temporal_nms_iou,
                minimum_uncovered_seconds=args.minimum_uncovered_seconds,
                halo_seconds=args.halo_seconds,
                low_confidence_margin=args.low_confidence_margin,
            )
            route_components = [component.__dict__ for component in route.components]
            selected = list(route.selected)
            confidence_margin = route.confidence_margin
            low_confidence_fallback = route.low_confidence_fallback
            retained_union_seconds = route.retained_union_seconds
        append_jsonl(routes_path, {
            'id': row['id'],
            'video': row['video'],
            'duration': float(row['duration']),
            'content_window_source': content_source,
            'candidates': [candidate.__dict__ for candidate in candidates],
            'temporal_policy': args.temporal_policy,
            'selected': selected,
            'components': route_components,
            'confidence_margin': confidence_margin,
            'low_confidence_fallback': low_confidence_fallback,
            'retained_union_seconds': retained_union_seconds,
            'query_embedding': query.tolist(),
        })
    del embedder
    gc.collect()


def _reshape_visual_tokens(output) -> tuple[np.ndarray, int, int, int]:
    grid = output.grid_thw.detach().cpu().tolist()
    if len(grid) != 1:
        raise ValueError('one component must produce one visual grid')
    temporal, height, width = (int(value) for value in grid[0])
    merged_h = height // output.spatial_merge_size
    merged_w = width // output.spatial_merge_size
    tokens = output.tokens.detach().float().cpu().numpy()
    return tokens.reshape(temporal, merged_h, merged_w, -1), temporal, merged_h, merged_w


def _aligned_rgb(frame_paths: list[str], count: int) -> list[np.ndarray]:
    import cv2

    selected = np.linspace(0, len(frame_paths) - 1, count).round().astype(int)
    output = []
    for index in selected:
        image = cv2.imread(frame_paths[int(index)])
        if image is None:
            raise RuntimeError(f'unable to read frame: {frame_paths[int(index)]}')
        output.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return output


def _pruning_config(args) -> SpatialPruningConfig:
    common = {
        'keep_ratio': 1.0 if args.spatial_policy == 'full' else args.spatial_keep_ratio,
        'minimum_tokens_per_frame': args.minimum_tokens_per_frame,
    }
    if args.spatial_policy in ('hybrid', 'random', 'full'):
        return SpatialPruningConfig(**common)
    if args.spatial_policy == 'query':
        return SpatialPruningConfig(
            **common,
            query_weight=1.0, motion_weight=0.0, uniqueness_weight=0.0, boundary_weight=0.0,
            query_quota=0.30, motion_quota=0.0, uniqueness_quota=0.0, boundary_quota=0.0,
        )
    if args.spatial_policy == 'motion':
        return SpatialPruningConfig(
            **common,
            query_weight=0.0, motion_weight=1.0, uniqueness_weight=0.0, boundary_weight=0.0,
            query_quota=0.0, motion_quota=0.30, uniqueness_quota=0.0, boundary_quota=0.0,
        )
    if args.spatial_policy == 'query-motion':
        return SpatialPruningConfig(
            **common,
            query_weight=0.5, motion_weight=0.5, uniqueness_weight=0.0, boundary_weight=0.0,
            query_quota=0.15, motion_quota=0.15, uniqueness_quota=0.0, boundary_quota=0.0,
        )
    if args.spatial_policy == 'query-motion-uniqueness':
        return SpatialPruningConfig(
            **common,
            query_weight=1 / 3, motion_weight=1 / 3, uniqueness_weight=1 / 3, boundary_weight=0.0,
            query_quota=0.10, motion_quota=0.10, uniqueness_quota=0.10, boundary_quota=0.0,
        )
    raise ValueError(f'unknown spatial policy: {args.spatial_policy}')


def _randomized_mask(
    pruning: SpatialPruningResult,
    *,
    seed_text: str,
    minimum_per_frame: int,
) -> SpatialPruningResult:
    shape = pruning.mask.shape
    total = int(np.prod(shape))
    seed = int.from_bytes(hashlib.sha256(seed_text.encode('utf-8')).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    kept = set()
    frame_size = shape[1] * shape[2]
    for frame in range(shape[0]):
        local = rng.choice(frame_size, size=min(minimum_per_frame, frame_size), replace=False)
        kept.update((frame * frame_size + local).tolist())
    remaining = pruning.budget - len(kept)
    candidates = np.asarray(sorted(set(range(total)) - kept), dtype=np.int64)
    if remaining > 0:
        kept.update(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False).tolist())
    indices = np.asarray(sorted(kept), dtype=np.int64)
    mask = np.zeros(total, dtype=bool)
    mask[indices] = True
    return SpatialPruningResult(
        mask=mask.reshape(shape),
        kept_indices=indices,
        signals=pruning.signals,
        budget=pruning.budget,
        protected_count=shape[0] * min(minimum_per_frame, frame_size),
    )


def spatial_phase(args, rows: list[dict], run_dir: Path, cache_root: Path) -> None:
    from vlmeval.vlm.qwen3_vl_embedding import Qwen3VLEmbedder

    routes = read_jsonl(run_dir / 'routes.jsonl', ('id',))
    trace_path = run_dir / 'spatial_traces.jsonl'
    complete = read_jsonl(trace_path, ('id', 'component'))
    embedder = Qwen3VLEmbedder(
        args.embedding_model,
        max_pixels=args.embedding_max_pixels,
        total_pixels=args.detailed_total_pixels,
        attn_implementation=args.attention,
    )
    pruning_config = _pruning_config(args)
    for row in rows:
        route = routes[(row['id'],)]
        components = route['components']
        allocations = distribute_frames(args.detailed_frame_budget, len(components))
        query = np.asarray(route['query_embedding'], dtype=np.float32)
        for component_index, (component, allocation) in enumerate(zip(components, allocations)):
            if (row['id'], component_index) in complete:
                continue
            start, end = float(component['start']), float(component['end'])
            timestamps = uniform_timestamps(start, end, allocation)
            frame_paths = extract_frames(
                row['video_path'],
                timestamps,
                cache_root / 'frames' / str(row['video']) / str(row['id']) / f'detail_{component_index}',
                max_width=args.frame_width,
            )
            visual = embedder.process_visual_tokens(
                frame_paths,
                sample_fps=len(frame_paths) / max(end - start, 1e-6),
            )
            features, token_frames, height, width = _reshape_visual_tokens(visual)
            token_timestamps = np.asarray(
                uniform_timestamps(start, end, token_frames), dtype=np.float64
            )
            rgb = _aligned_rgb(frame_paths, token_frames)
            pruning = prune_spatial_tokens(
                features,
                query,
                [component_index] * token_frames,
                rgb_frames=rgb,
                config=pruning_config,
            )
            if args.spatial_policy == 'random':
                pruning = _randomized_mask(
                    pruning,
                    seed_text=f'{row["id"]}:{component_index}:{args.run_name}',
                    minimum_per_frame=args.minimum_tokens_per_frame,
                )
            artifact = run_dir / 'spatial' / str(row['id']) / f'{component_index}.npz'
            atomic_npz(
                artifact,
                timestamps=token_timestamps,
                features=features.astype(np.float16),
                mask=pruning.mask,
                query=pruning.signals.query,
                motion=pruning.signals.motion,
                uniqueness=pruning.signals.uniqueness,
                boundary=pruning.signals.boundary,
                combined=pruning.signals.combined,
                motion_fallback=pruning.signals.motion_fallback,
            )
            append_jsonl(trace_path, {
                'id': row['id'],
                'component': component_index,
                'component_window': [start, end],
                'component_score': float(component['score']),
                'spatial_policy': args.spatial_policy,
                'frame_paths': frame_paths,
                'decoded_frames': len(frame_paths),
                'token_frames': token_frames,
                'grid': [height, width],
                'dense_tokens': int(features.shape[0] * features.shape[1] * features.shape[2]),
                'retained_tokens': int(pruning.mask.sum()),
                'protected_tokens': pruning.protected_count,
                'artifact': str(artifact),
            })
    del embedder
    gc.collect()


def _proposal_prediction(row: dict, traces: list[dict], args) -> list[list[float]]:
    proposals = []
    config = ProposalScorerConfig(maximum_proposals=args.maximum_output_intervals)
    for trace in traces:
        with np.load(trace['artifact'], allow_pickle=False) as value:
            evidence = evidence_curve(value['query'], value['motion'], value['mask'])
            component_proposals = propose_intervals(
                value['timestamps'],
                evidence,
                cardinality='multi',
                route_score=float(trace['component_score']),
                config=config,
            )
        proposals.extend(component_proposals)
    proposals.sort(key=lambda item: (-item.score, item.start, item.end))
    if row['cardinality'] == 'single':
        proposals = proposals[:1]
    else:
        proposals = proposals[:args.maximum_output_intervals]
    return consolidate_intervals(
        [[item.start, item.end] for item in proposals],
        duration=float(row['duration']),
    )


def _controlled_prompt(query: str, duration: float, cardinality: str, local: bool) -> str:
    scope = 'relative to the start of this clip' if local else 'relative to the full video'
    task = 'Find every disjoint time interval' if cardinality == 'multi' else 'Find the single best time interval'
    return (
        f'{task} where this event occurs: {query!r}. Timestamps must be in seconds {scope}, '
        f'between 0 and {duration:.3f}. Return only a JSON array of [start, end] pairs.'
    )


def _timelens_prediction(row: dict, traces: list[dict], args, model) -> list[list[float]]:
    candidates = []
    for trace in traces:
        start, end = trace['component_window']
        duration = end - start
        message = [
            {
                'type': 'video',
                'value': trace['frame_paths'],
                'sample_fps': len(trace['frame_paths']) / max(duration, 1e-6),
            },
            {'type': 'text', 'value': _controlled_prompt(row['query'], duration, row['cardinality'], True)},
        ]
        response = model.generate(message, dataset='HybridVTG')
        for local_start, local_end in parse_time_intervals(response):
            local_start = max(0.0, float(local_start))
            local_end = min(duration, float(local_end))
            if local_end > local_start:
                candidates.append([local_start + start, local_end + start])
    merged = consolidate_intervals(candidates, duration=float(row['duration']))
    if row['cardinality'] == 'single' and merged:
        return [max(merged, key=lambda interval: interval[1] - interval[0])]
    return merged


def ground_phase(args, rows: list[dict], run_dir: Path) -> None:
    traces = read_jsonl(run_dir / 'spatial_traces.jsonl', ('id', 'component'))
    predictions_path = run_dir / 'predictions.jsonl'
    complete = read_jsonl(predictions_path, ('id', 'backend'))
    model = None
    if args.grounder_backend == 'timelens-frames':
        from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat

        model = Qwen3VLChat(
            model_path=args.model,
            use_custom_prompt=False,
            use_vllm=False,
            use_audio_in_video=False,
            max_new_tokens=512,
            temperature=0.01,
            top_p=0.001,
            top_k=1,
            fps=None,
            nframe=None,
            min_pixels=32 * 32,
            max_pixels=args.frame_width * args.frame_width,
            total_pixels=args.detailed_total_pixels,
            attn_implementation=args.attention,
        )
        model.model.eval()
        model.model.requires_grad_(False)
    for row in rows:
        key = (row['id'], args.grounder_backend)
        if key in complete:
            continue
        row_traces = sorted(
            [value for (row_id, _), value in traces.items() if row_id == row['id']],
            key=lambda value: value['component'],
        )
        if args.grounder_backend == 'proposal':
            prediction = _proposal_prediction(row, row_traces, args)
        else:
            prediction = _timelens_prediction(row, row_traces, args, model)
        append_jsonl(predictions_path, {
            'id': row['id'],
            'video': row['video'],
            'duration': float(row['duration']),
            'query': row['query'],
            'backend': args.grounder_backend,
            'prediction': prediction,
            'refined_prediction': prediction,
            'decoded_frames': sum(value['decoded_frames'] for value in row_traces),
            'dense_tokens': sum(value['dense_tokens'] for value in row_traces),
            'retained_tokens': sum(value['retained_tokens'] for value in row_traces),
            'grounder_calls': len(row_traces) if model is not None else 0,
        })
    del model
    gc.collect()


def refine_phase(args, rows: list[dict], run_dir: Path) -> None:
    from vlmeval.vlm.qwen3_vl_embedding import Qwen3VLEmbedder

    predictions_path = run_dir / 'predictions.jsonl'
    predictions = read_jsonl(predictions_path, ('id', 'backend'))
    routes = read_jsonl(run_dir / 'routes.jsonl', ('id',))
    output_path = run_dir / 'refinements.jsonl'
    complete = read_jsonl(output_path, ('id', 'backend'))
    config = BoundaryRefinementConfig(
        radius_seconds=args.refinement_radius,
        evidence_window_seconds=args.refinement_window,
        continuity_weight=args.refinement_continuity_weight,
        minimum_gain=args.refinement_minimum_gain,
    )
    embedder = Qwen3VLEmbedder(
        args.embedding_model,
        max_pixels=args.embedding_max_pixels,
        total_pixels=args.refinement_total_pixels,
        attn_implementation=args.attention,
    )

    def local_trace(row: dict, query: np.ndarray, center: float, label: str):
        start = max(0.0, center - args.refinement_radius)
        end = min(float(row['duration']), center + args.refinement_radius)
        count = max(2, int(math.ceil((end - start) * args.refinement_fps)))
        timestamps = uniform_timestamps(start, end, count)
        frame_paths = extract_frames(
            row['video_path'],
            timestamps,
            Path(args.cache_root) / args.run_name / args.dataset / 'frames'
            / str(row['video']) / str(row['id']) / label,
            max_width=args.frame_width,
        )
        visual = embedder.process_visual_tokens(
            frame_paths,
            sample_fps=len(frame_paths) / max(end - start, 1e-6),
        )
        features, token_frames, _, _ = _reshape_visual_tokens(visual)
        rgb = _aligned_rgb(frame_paths, token_frames)
        query_scores = query_relevance(features, query)
        motion_scores, _ = motion_importance(features, [0] * token_frames, rgb)
        keep = np.ones(query_scores.shape, dtype=bool)
        evidence = evidence_curve(query_scores, motion_scores, keep)
        continuity = motion_scores.mean(axis=(1, 2))
        token_timestamps = np.asarray(uniform_timestamps(start, end, token_frames), dtype=float)
        return token_timestamps, evidence, continuity, len(frame_paths), int(np.prod(query_scores.shape))

    for row in rows:
        key = (row['id'], args.grounder_backend)
        if key in complete:
            continue
        record = predictions[key]
        query = np.asarray(routes[(row['id'],)]['query_embedding'], dtype=np.float32)
        refined = []
        audits = []
        refinement_frames = 0
        refinement_tokens = 0
        components = routes[(row['id'],)]['components']
        for interval_index, interval in enumerate(record['prediction']):
            component = next((
                [value['start'], value['end']] for value in components
                if value['start'] <= interval[0] and value['end'] >= interval[1]
            ), [0.0, float(row['duration'])])
            start_trace = local_trace(row, query, float(interval[0]), f'refine_{interval_index}_start')
            start_result = refine_interval(
                interval,
                start_trace[0],
                start_trace[1],
                start_trace[2],
                duration=float(row['duration']),
                component=component,
                config=config,
            )
            end_trace = local_trace(row, query, float(interval[1]), f'refine_{interval_index}_end')
            end_result = refine_interval(
                start_result.refined,
                end_trace[0],
                end_trace[1],
                end_trace[2],
                duration=float(row['duration']),
                component=component,
                config=config,
            )
            final_interval = [start_result.refined[0], end_result.refined[1]]
            if final_interval[1] <= final_interval[0]:
                final_interval = list(interval)
            refined.append(final_interval)
            audits.append({
                'start': start_result.__dict__,
                'end': end_result.__dict__,
            })
            refinement_frames += start_trace[3] + end_trace[3]
            refinement_tokens += start_trace[4] + end_trace[4]
        refined = consolidate_intervals(refined, duration=float(row['duration']))
        append_jsonl(output_path, {
            'id': row['id'],
            'backend': args.grounder_backend,
            'prediction': record['prediction'],
            'refined_prediction': refined,
            'audits': audits,
            'refinement_frames': refinement_frames,
            'refinement_dense_tokens': refinement_tokens,
        })
    del embedder
    gc.collect()


def evaluate_phase(args, rows: list[dict], run_dir: Path) -> dict:
    predictions = read_jsonl(run_dir / 'predictions.jsonl', ('id', 'backend'))
    refinements = read_jsonl(run_dir / 'refinements.jsonl', ('id', 'backend'))
    records = []
    for row in rows:
        key = (row['id'], args.grounder_backend)
        if key not in predictions:
            continue
        prediction = refinements.get(key, predictions[key]).get(
            'refined_prediction', predictions[key]['prediction']
        )
        refinement = refinements.get(key, {})
        metrics = per_sample_metrics(prediction, row['targets'])
        records.append({
            'id': row['id'],
            'video': row['video'],
            **metrics,
            'decoded_frames': predictions[key]['decoded_frames'],
            'dense_tokens': predictions[key]['dense_tokens'],
            'retained_tokens': predictions[key]['retained_tokens'],
            'refinement_frames': int(refinement.get('refinement_frames', 0)),
            'refinement_dense_tokens': int(refinement.get('refinement_dense_tokens', 0)),
        })
    summary = {
        'dataset': args.dataset,
        'split': args.split,
        'backend': args.grounder_backend,
        'samples': len(records),
        'complete': len(records) == len(rows),
    }
    if records:
        for metric in ('mIoU', 'R@0.1', 'R@0.2', 'R@0.3', 'R@0.5', 'R@0.7'):
            summary[metric] = 100 * float(np.mean([row[metric] for row in records]))
        summary['CardinalityError'] = float(np.mean([row['CardinalityError'] for row in records]))
        for metric in (
            'decoded_frames', 'dense_tokens', 'retained_tokens',
            'refinement_frames', 'refinement_dense_tokens',
        ):
            summary[metric] = int(sum(row[metric] for row in records))
        summary['total_frames'] = summary['decoded_frames'] + summary['refinement_frames']
        summary['total_scored_tokens'] = summary['dense_tokens'] + summary['refinement_dense_tokens']
        summary['token_retention'] = (
            summary['retained_tokens'] / summary['dense_tokens'] if summary['dense_tokens'] else 0.0
        )
    result = {'summary': summary, 'records': records}
    atomic_json(run_dir / 'summary.json', result)
    pd.DataFrame([summary]).to_csv(run_dir / 'summary.csv', index=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--split', default='auto')
    parser.add_argument('--phase', choices=(
        'validate', 'index', 'route', 'spatial', 'ground', 'refine', 'evaluate', 'all'
    ), default='all')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/timelens2-hybrid'))
    parser.add_argument('--run-name', default='smoke')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--model', default='MCG-NJU/TimeLens2-4B')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-VL-Embedding-2B')
    parser.add_argument('--grounder-backend', choices=('proposal', 'timelens-frames'), default='proposal')
    parser.add_argument('--coarse-fps', type=float, default=1.0)
    parser.add_argument('--window-scales', default='8,16,32,64')
    parser.add_argument('--window-stride-ratio', type=float, default=0.5)
    parser.add_argument('--mean-score-weight', type=float, default=0.5)
    parser.add_argument(
        '--temporal-policy',
        choices=('hybrid', 'content-only', 'uniform-full'),
        default='hybrid',
    )
    parser.add_argument('--temporal-union-budget', type=float, default=32.0)
    parser.add_argument('--maximum-candidates', type=int, default=8)
    parser.add_argument('--temporal-nms-iou', type=float, default=0.7)
    parser.add_argument('--minimum-uncovered-seconds', type=float, default=1.0)
    parser.add_argument('--halo-seconds', type=float, default=2.0)
    parser.add_argument('--low-confidence-margin', type=float, default=0.05)
    parser.add_argument('--detailed-frame-budget', type=int, default=128)
    parser.add_argument('--spatial-keep-ratio', type=float, default=0.25)
    parser.add_argument(
        '--spatial-policy',
        choices=(
            'hybrid', 'full', 'random', 'query', 'motion',
            'query-motion', 'query-motion-uniqueness',
        ),
        default='hybrid',
    )
    parser.add_argument('--minimum-tokens-per-frame', type=int, default=1)
    parser.add_argument('--maximum-output-intervals', type=int, default=8)
    parser.add_argument('--refinement-radius', type=float, default=2.0)
    parser.add_argument('--refinement-fps', type=float, default=8.0)
    parser.add_argument('--refinement-window', type=float, default=0.5)
    parser.add_argument('--refinement-continuity-weight', type=float, default=0.25)
    parser.add_argument('--refinement-minimum-gain', type=float, default=0.01)
    parser.add_argument('--frame-width', type=int, default=336)
    parser.add_argument('--embedding-max-pixels', type=int, default=336 * 336)
    parser.add_argument('--detailed-total-pixels', type=int, default=128 * 336 * 336)
    parser.add_argument('--refinement-total-pixels', type=int, default=64 * 336 * 336)
    parser.add_argument('--embedding-batch-size', type=int, default=8)
    parser.add_argument('--attention', default='sdpa')
    parser.add_argument('--keep-frame-cache', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.window_scales = parse_csv(args.window_scales, float)
    if args.coarse_fps <= 0 or args.detailed_frame_budget <= 0:
        raise ValueError('FPS and frame budget must be positive')
    if not 0 < args.spatial_keep_ratio <= 1:
        raise ValueError('spatial keep ratio must be in (0, 1]')
    rows, metadata, _ = load_rows(args)
    run_dir = args.output_root / args.run_name / args.dataset
    cache_root = args.cache_root / args.run_name / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'metadata': metadata, 'rows': rows}
    manifest_path = run_dir / 'dataset_manifest.json'
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + '\n'
    if manifest_path.is_file() and manifest_path.read_text(encoding='utf-8') != manifest_text:
        raise RuntimeError('dataset manifest changed; use a new run name')
    manifest_path.write_text(manifest_text, encoding='utf-8')
    config = {
        key: value for key, value in vars(args).items()
        if key not in ('phase', 'keep_frame_cache')
    }
    config_path = run_dir / 'config.json'
    config_text = json.dumps(config, indent=2, sort_keys=True, default=_json_default) + '\n'
    if config_path.is_file() and config_path.read_text(encoding='utf-8') != config_text:
        raise RuntimeError('run configuration changed; use a new run name')
    config_path.write_text(config_text, encoding='utf-8')
    append_jsonl(run_dir / 'invocations.jsonl', {
        'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'phase': args.phase,
    })
    if args.phase in ('validate', 'all'):
        validate_inputs(rows)
    if args.phase in ('index', 'all'):
        index_phase(args, rows, run_dir, cache_root)
    if args.phase in ('route', 'all'):
        route_phase(args, rows, run_dir)
    if args.phase in ('spatial', 'all'):
        spatial_phase(args, rows, run_dir, cache_root)
    if args.phase in ('ground', 'all'):
        ground_phase(args, rows, run_dir)
    if args.phase in ('refine', 'all'):
        refine_phase(args, rows, run_dir)
    if args.phase in ('evaluate', 'all'):
        evaluate_phase(args, rows, run_dir)
    if args.phase == 'all' and not args.keep_frame_cache:
        frame_cache = cache_root / 'frames'
        if frame_cache.is_dir():
            shutil.rmtree(frame_cache)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
