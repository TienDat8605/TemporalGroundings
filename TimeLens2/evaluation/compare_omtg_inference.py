#!/usr/bin/env python3
"""Compile multi-model OMTG inference results with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vlmeval.dataset.omtgbench import compute_one_to_many_metrics, parse_time_intervals

METRICS = (
    'C-Acc',
    'tIoU',
    'CardinalityError',
    'tP@0.3',
    'tR@0.3',
    'tF1@0.3',
    'tP@0.5',
    'tR@0.5',
    'tF1@0.5',
    'tP@0.7',
    'tR@0.7',
    'tF1@0.7',
    'EtF1',
)
REPORT_METRICS = ('C-Acc', 'EtF1', 'tIoU', 'tF1@0.3', 'tF1@0.5', 'tF1@0.7')
PERCENT_METRICS = frozenset(METRICS) - {'CardinalityError'}
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_725
SUMMARY_TOLERANCE = 1e-8
MODEL_LABELS = {
    'MCG-NJU/TimeLens2-4B': 'TimeLens2-4B',
    'Qwen/Qwen3-VL-4B-Instruct': 'Qwen3-VL-4B',
}


@dataclass
class Experiment:
    """One model and inference-family result bundle."""

    family: str
    model: str
    summary_path: Path
    results: dict[str, dict]
    per_query: dict[str, dict[int, dict[str, float]]]
    answers: dict[int, str]
    questions: dict[int, str]
    videos: dict[int, str]
    durations: dict[int, float]
    prediction_counts: dict[str, dict[int, int]]


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
        raise ValueError(f'Invalid OMTG summary: {path}')
    return data


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing OMTG predictions: {path}')
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSONL at {path}:{line_number}') from exc
        if not isinstance(row, dict):
            raise ValueError(f'Expected an object at {path}:{line_number}')
        rows.append(row)
    return rows


def result_key(row: dict) -> str:
    schedule = str(row['schedule'])
    if 'budget' in row:
        return f'{schedule}-{int(row["budget"])}f'
    return schedule


def record_result_key(record: dict, family: str) -> str:
    if family == 'paper-style-2fps':
        return f'paper-2fps-{record["prompt_mode"]}'
    return f'{record["schedule"]}-{int(record["budget"])}f'


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def model_slug(model: str) -> str:
    label = model_label(model).lower()
    return re.sub(r'[^a-z0-9]+', '-', label).strip('-')


def model_sort_key(model: str) -> tuple[int, str]:
    priority = {
        'MCG-NJU/TimeLens2-4B': 0,
        'Qwen/Qwen3-VL-4B-Instruct': 1,
    }
    return priority.get(model, 2), model_label(model)


def scaled_metrics(prediction: list[list[float]], answer: str) -> dict[str, float]:
    metrics = compute_one_to_many_metrics(prediction, parse_time_intervals(str(answer)))
    return {
        metric: float(value) * (100.0 if metric in PERCENT_METRICS else 1.0)
        for metric, value in metrics.items()
    }


def prediction_for_record(record: dict, family: str) -> list[list[float]]:
    if family == 'paper-style-2fps' and 'response' in record:
        return parse_time_intervals(str(record['response']))
    prediction = record.get('prediction', [])
    if not isinstance(prediction, list):
        raise ValueError(f'Invalid prediction for query {record.get("id")}')
    return prediction


def aggregate_per_query(rows: dict[int, dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        metric: float(np.mean([row[metric] for row in rows.values()]))
        for metric in METRICS
    }


def validate_summary(
    summary_path: Path,
    summary_results: dict[str, dict],
    per_query: dict[str, dict[int, dict[str, float]]],
) -> None:
    for key, stored in summary_results.items():
        if key not in per_query:
            raise ValueError(f'No predictions for {key!r} in {summary_path.parent}')
        observed = aggregate_per_query(per_query[key])
        for metric in METRICS:
            if metric not in stored:
                continue
            if not np.isclose(
                float(stored[metric]),
                observed[metric],
                rtol=0.0,
                atol=SUMMARY_TOLERANCE,
            ):
                raise ValueError(
                    f'{summary_path}: stored {key} {metric}={stored[metric]} '
                    f'does not match rescored value {observed[metric]}'
                )


def load_experiment(summary_path: Path, family: str) -> Experiment:
    summary = load_summary(summary_path)
    model = str(summary.get('model') or '')
    if not model:
        raise ValueError(f'Missing model in {summary_path}')
    incomplete = [
        result_key(row) for row in summary['results'] if not row.get('complete', False)
    ]
    if incomplete:
        raise ValueError(f'Cannot compare incomplete results in {summary_path}: {incomplete}')

    summary_results = {result_key(row): row for row in summary['results']}
    per_query: dict[str, dict[int, dict[str, float]]] = {}
    prediction_counts: dict[str, dict[int, int]] = {}
    answers: dict[int, str] = {}
    questions: dict[int, str] = {}
    videos: dict[int, str] = {}
    durations: dict[int, float] = {}
    for record in load_jsonl(summary_path.parent / 'predictions.jsonl'):
        key = record_result_key(record, family)
        query_id = int(record['id'])
        if query_id in per_query.setdefault(key, {}):
            raise ValueError(f'Duplicate prediction for {key}, query {query_id}')
        answer = str(record['answer'])
        question = str(record['question'])
        video = str(record.get('video', f'query-{query_id}'))
        if query_id in answers and answers[query_id] != answer:
            raise ValueError(f'Conflicting answer for query {query_id} in {summary_path.parent}')
        if query_id in questions and questions[query_id] != question:
            raise ValueError(f'Conflicting question for query {query_id} in {summary_path.parent}')
        prediction = prediction_for_record(record, family)
        answers[query_id] = answer
        questions[query_id] = question
        if query_id in videos and videos[query_id] != video:
            raise ValueError(f'Conflicting video for query {query_id} in {summary_path.parent}')
        videos[query_id] = video
        if record.get('duration') is not None:
            durations[query_id] = float(record['duration'])
        per_query[key][query_id] = scaled_metrics(prediction, answer)
        prediction_counts.setdefault(key, {})[query_id] = len(prediction)

    validate_summary(summary_path, summary_results, per_query)
    return Experiment(
        family=family,
        model=model,
        summary_path=summary_path,
        results=summary_results,
        per_query=per_query,
        answers=answers,
        questions=questions,
        videos=videos,
        durations=durations,
        prediction_counts=prediction_counts,
    )


def normalized_result(row: dict, family: str, model: str | None = None) -> dict:
    output = dict(row, family=family)
    output['result'] = result_key(row)
    if model is not None:
        output['model'] = model
        output['model_label'] = model_label(model)
        output['result_id'] = f'{model_slug(model)}/{output["result"]}'
    output['input_frames'] = row.get('total_frames', row.get('requested_frames_estimate'))
    output['frame_count_kind'] = (
        'measured' if row.get('total_frames') is not None else 'duration-derived estimate'
    )
    output['synchronized_model_seconds'] = row.get('gpu_seconds')
    output['standalone_wall_seconds'] = row.get('wall_seconds')
    output['peak_vram_gib'] = (
        float(row['peak_vram_bytes']) / (1024 ** 3)
        if row.get('peak_vram_bytes') is not None else None
    )
    output['timing_note'] = (
        'router preprocessing and model latency included'
        if row.get('schedule') in {
            'embedding-window-local', 'score-window-local',
            'residual-window-local', 'residual-window-local-no-stop',
        }
        else 'historical gpu_seconds relabeled as synchronized model-call latency'
    )
    return output


def metric_deltas(candidate: dict, reference: dict) -> dict:
    return {
        f'delta_{metric}': float(candidate[metric]) - float(reference[metric])
        for metric in METRICS
        if metric in candidate and metric in reference
    }


def paired_bootstrap_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if candidate.shape != reference.shape:
        raise ValueError('Paired bootstrap arrays must have the same shape')
    if candidate.ndim != 1 or not len(candidate):
        raise ValueError('Paired bootstrap requires non-empty one-dimensional arrays')
    if samples <= 0:
        raise ValueError('Bootstrap samples must be positive')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(samples, len(candidate)))
    deltas = (candidate[indices] - reference[indices]).mean(axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def paired_cluster_bootstrap_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    clusters: list[str],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if candidate.shape != reference.shape or len(candidate) != len(clusters):
        raise ValueError('Clustered bootstrap inputs must have equal length')
    if candidate.ndim != 1 or not len(candidate):
        raise ValueError('Clustered bootstrap requires non-empty one-dimensional arrays')
    unique = sorted(set(clusters))
    cluster_index = {cluster: index for index, cluster in enumerate(unique)}
    sums = np.zeros(len(unique))
    counts = np.zeros(len(unique))
    for delta, cluster in zip(candidate - reference, clusters):
        index = cluster_index[cluster]
        sums[index] += delta
        counts[index] += 1
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(unique), size=(samples, len(unique)))
    estimates = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def comparison_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode('utf-8')).digest()
    return (base_seed + int.from_bytes(digest[:4], 'big')) % (2 ** 32)


def validate_pair(candidate: Experiment, reference: Experiment, candidate_key: str, reference_key: str) -> list[int]:
    candidate_ids = set(candidate.per_query[candidate_key])
    reference_ids = set(reference.per_query[reference_key])
    if candidate_ids != reference_ids:
        raise ValueError(
            f'Query ID mismatch between {candidate.model}/{candidate_key} and '
            f'{reference.model}/{reference_key}'
        )
    for query_id in candidate_ids:
        if candidate.answers[query_id] != reference.answers[query_id]:
            raise ValueError(f'Answer mismatch for paired query {query_id}')
        if candidate.videos[query_id] != reference.videos[query_id]:
            raise ValueError(f'Video mismatch for paired query {query_id}')
    return sorted(candidate_ids)


def build_paired_comparison(
    name: str,
    comparison_class: str,
    candidate: Experiment,
    candidate_key: str,
    reference: Experiment,
    reference_key: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    query_ids = validate_pair(candidate, reference, candidate_key, reference_key)
    candidate_summary = normalized_result(
        candidate.results[candidate_key], candidate.family, candidate.model
    )
    reference_summary = normalized_result(
        reference.results[reference_key], reference.family, reference.model
    )
    output = {
        'name': name,
        'comparison_class': comparison_class,
        'candidate': candidate_summary['result_id'],
        'reference': reference_summary['result_id'],
        'samples': len(query_ids),
        'synchronized_model_latency_ratio': safe_ratio(
            candidate_summary.get('synchronized_model_seconds'),
            reference_summary.get('synchronized_model_seconds'),
        ),
        'standalone_wall_ratio': safe_ratio(
            candidate_summary.get('standalone_wall_seconds'),
            reference_summary.get('standalone_wall_seconds'),
        ),
        'input_frame_ratio': safe_ratio(
            candidate_summary.get('input_frames'),
            reference_summary.get('input_frames'),
        ),
        **metric_deltas(candidate_summary, reference_summary),
    }
    seed = comparison_seed(bootstrap_seed, name)
    for metric in METRICS:
        candidate_values = np.asarray([
            candidate.per_query[candidate_key][query_id][metric] for query_id in query_ids
        ])
        reference_values = np.asarray([
            reference.per_query[reference_key][query_id][metric] for query_id in query_ids
        ])
        low, high = paired_cluster_bootstrap_interval(
            candidate_values,
            reference_values,
            [candidate.videos[query_id] for query_id in query_ids],
            bootstrap_samples,
            seed,
        )
        output[f'ci95_low_{metric}'] = low
        output[f'ci95_high_{metric}'] = high
    return output


def safe_ratio(candidate, reference):
    if candidate is None or reference in (None, 0):
        return None
    return float(candidate) / float(reference)


def indexed_experiments(experiments: list[Experiment]) -> dict[tuple[str, str], Experiment]:
    index = {}
    for experiment in experiments:
        key = (experiment.model, experiment.family)
        if key in index:
            raise ValueError(f'Duplicate {experiment.family} experiment for {experiment.model}')
        index[key] = experiment
    return index


def add_if_available(
    comparisons: list[dict],
    index: dict[tuple[str, str], Experiment],
    *,
    name: str,
    comparison_class: str,
    candidate_model: str,
    candidate_family: str,
    candidate_key: str,
    reference_model: str,
    reference_family: str,
    reference_key: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    candidate = index.get((candidate_model, candidate_family))
    reference = index.get((reference_model, reference_family))
    if not candidate or not reference:
        return
    if candidate_key not in candidate.results or reference_key not in reference.results:
        return
    comparisons.append(build_paired_comparison(
        name=name,
        comparison_class=comparison_class,
        candidate=candidate,
        candidate_key=candidate_key,
        reference=reference,
        reference_key=reference_key,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    ))


def build_comparisons(
    experiments: list[Experiment],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    index = indexed_experiments(experiments)
    models = sorted({experiment.model for experiment in experiments}, key=model_sort_key)
    comparisons: list[dict] = []
    for model in models:
        slug = model_slug(model)
        add_if_available(
            comparisons, index,
            name=f'{slug}:controlled-vs-official-2fps',
            comparison_class='prompt-ablation',
            candidate_model=model, candidate_family='paper-style-2fps',
            candidate_key='paper-2fps-controlled',
            reference_model=model, reference_family='paper-style-2fps',
            reference_key='paper-2fps-official',
            bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
        )
        search = index.get((model, 'fixed-frame'))
        baseline = index.get((model, 'paper-style-2fps'))
        if search:
            budgets = sorted({
                int(row['budget']) for row in search.results.values() if 'budget' in row
            })
            for budget in budgets:
                uniform_key = f'uniform-one-shot-{budget}f'
                for candidate_key in search.results:
                    row = search.results[candidate_key]
                    if int(row.get('budget', -1)) != budget or candidate_key == uniform_key:
                        continue
                    add_if_available(
                        comparisons, index,
                        name=f'{slug}:{candidate_key}-vs-{uniform_key}',
                        comparison_class='fixed-budget-schedule-ablation',
                        candidate_model=model, candidate_family='fixed-frame',
                        candidate_key=candidate_key,
                        reference_model=model, reference_family='fixed-frame',
                        reference_key=uniform_key,
                        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
                    )
            schedules = sorted({
                str(row['schedule']) for row in search.results.values()
                if int(row.get('budget', -1)) == 64
            })
            for schedule in schedules:
                add_if_available(
                    comparisons, index,
                    name=f'{slug}:{schedule}-128f-vs-64f',
                    comparison_class='budget-scaling',
                    candidate_model=model, candidate_family='fixed-frame',
                    candidate_key=f'{schedule}-128f',
                    reference_model=model, reference_family='fixed-frame',
                    reference_key=f'{schedule}-64f',
                    bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
                )
        if search and baseline:
            for budget in (64, 128):
                add_if_available(
                    comparisons, index,
                    name=f'{slug}:embedding-{budget}f-vs-controlled-2fps',
                    comparison_class='input-policy-tradeoff',
                    candidate_model=model, candidate_family='fixed-frame',
                    candidate_key=f'embedding-window-local-{budget}f',
                    reference_model=model, reference_family='paper-style-2fps',
                    reference_key='paper-2fps-controlled',
                    bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
                )

    timelens = 'MCG-NJU/TimeLens2-4B'
    qwen = 'Qwen/Qwen3-VL-4B-Instruct'
    for family, key in (
        ('paper-style-2fps', 'paper-2fps-controlled'),
        ('fixed-frame', 'uniform-one-shot-64f'),
        ('fixed-frame', 'embedding-window-local-64f'),
    ):
        add_if_available(
            comparisons, index,
            name=f'timelens2-vs-qwen:{key}',
            comparison_class='model-comparison',
            candidate_model=timelens, candidate_family=family, candidate_key=key,
            reference_model=qwen, reference_family=family, reference_key=key,
            bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
        )
    return comparisons


def prompt_audit(experiments: list[Experiment]) -> dict:
    baselines = [
        experiment for experiment in experiments
        if experiment.family == 'paper-style-2fps'
    ]
    if not baselines:
        return {}
    source = baselines[0]
    query_ids = sorted(source.questions)
    singular = sum(
        source.questions[query_id].startswith('Find the video segment ')
        and 'determine its start and end seconds.' in source.questions[query_id]
        for query_id in query_ids
    )
    target_counts = [
        len(parse_time_intervals(source.answers[query_id])) for query_id in query_ids
    ]
    behavior = []
    for experiment in baselines:
        for key in ('paper-2fps-official', 'paper-2fps-controlled'):
            counts = list(experiment.prediction_counts.get(key, {}).values())
            if not counts:
                continue
            behavior.append({
                'model': experiment.model,
                'result': key,
                'mean_predicted_spans': float(np.mean(counts)),
                'empty_outputs': sum(count == 0 for count in counts),
                'single_span_outputs': sum(count == 1 for count in counts),
                'multi_span_outputs': sum(count > 1 for count in counts),
            })
    return {
        'samples': len(query_ids),
        'singular_instruction_samples': singular,
        'multi_span_label_samples': sum(count > 1 for count in target_counts),
        'minimum_label_spans': min(target_counts),
        'maximum_label_spans': max(target_counts),
        'mean_label_spans': float(np.mean(target_counts)),
        'output_behavior': behavior,
    }


def stratified_results(experiments: list[Experiment]) -> list[dict]:
    """Produce descriptive, fixed-bin slices without inferential claims."""
    durations = {}
    for experiment in experiments:
        durations.update(experiment.durations)
    rows = []
    for experiment in experiments:
        for key, metrics_by_id in experiment.per_query.items():
            groups: dict[tuple[str, str], list[int]] = {}
            for query_id in metrics_by_id:
                if query_id not in durations:
                    continue
                spans = parse_time_intervals(experiment.answers[query_id])
                duration = durations[query_id]
                lengths = sorted(end - start for start, end in spans)
                centers = [(start + end) / 2 for start, end in spans]
                density = sum(lengths) / max(duration, 1e-9)
                dispersion = (max(centers) - min(centers)) / max(duration, 1e-9)
                values = {
                    'duration': '≤120s' if duration <= 120 else ('120–300s' if duration <= 300 else '>300s'),
                    'span_count': '2–3' if len(spans) <= 3 else ('4–5' if len(spans) <= 5 else '≥6'),
                    'median_span': '≤5s' if lengths[len(lengths) // 2] <= 5 else (
                        '5–15s' if lengths[len(lengths) // 2] <= 15 else '>15s'
                    ),
                    'evidence_density': '<10%' if density < 0.1 else ('10–30%' if density < 0.3 else '≥30%'),
                    'dispersion': '<25%' if dispersion < 0.25 else ('25–60%' if dispersion < 0.6 else '≥60%'),
                }
                for dimension, bucket in values.items():
                    groups.setdefault((dimension, bucket), []).append(query_id)
            for (dimension, bucket), query_ids in sorted(groups.items()):
                rows.append({
                    'model': experiment.model,
                    'result': key,
                    'dimension': dimension,
                    'bucket': bucket,
                    'samples': len(query_ids),
                    **{
                        metric: float(np.mean([
                            metrics_by_id[query_id][metric] for query_id in query_ids
                        ]))
                        for metric in ('C-Acc', 'EtF1', 'tIoU')
                    },
                })
    return rows


def build_multi_model_comparison(
    baseline_paths: list[Path],
    search_paths: list[Path],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    experiments = [
        *(load_experiment(path, 'paper-style-2fps') for path in baseline_paths),
        *(load_experiment(path, 'fixed-frame') for path in search_paths),
    ]
    results = [
        normalized_result(row, experiment.family, experiment.model)
        for experiment in experiments
        for row in experiment.results.values()
    ]
    return {
        'dataset': 'OMTGBench',
        'models': sorted({experiment.model for experiment in experiments}, key=model_sort_key),
        'primary_metric': 'EtF1',
        'bootstrap': {
            'method': 'paired percentile bootstrap clustered by video',
            'samples': bootstrap_samples,
            'seed': bootstrap_seed,
            'confidence_level': 0.95,
            'multiple_comparison_correction': False,
        },
        'compute_policy': (
            'gpu_seconds is synchronized end-to-end model-call latency, not CUDA kernel time. '
            'Wall time includes decoding, routing, Python, processor, and file overhead. '
            'Runtime values are descriptive because every setting has only one execution.'
        ),
        'prompt_audit': prompt_audit(experiments),
        'stratified_results': stratified_results(experiments),
        'results': results,
        'comparisons': build_comparisons(experiments, bootstrap_samples, bootstrap_seed),
    }


def build_comparison(baseline: dict, search: dict) -> dict:
    """Preserve the original summary-only single-model API."""
    baseline_model = baseline.get('model')
    search_model = search.get('model')
    if baseline_model and search_model and baseline_model != search_model:
        raise ValueError(
            f'Model mismatch: baseline uses {baseline_model!r}, search uses {search_model!r}'
        )
    incomplete = [
        result_key(row)
        for row in baseline['results'] + search['results']
        if not row.get('complete', False)
    ]
    if incomplete:
        raise ValueError(f'Cannot compare incomplete results: {incomplete}')
    baseline_rows = [normalized_result(row, 'paper-style-2fps') for row in baseline['results']]
    search_rows = [normalized_result(row, 'fixed-frame') for row in search['results']]
    all_rows = baseline_rows + search_rows
    keyed = {result_key(row): row for row in all_rows}
    comparisons = []
    official = keyed.get('paper-2fps-official')
    controlled = keyed.get('paper-2fps-controlled')
    if official and controlled:
        comparisons.append({
            'name': 'controlled-prompt-vs-official-prompt-at-2fps',
            'candidate': result_key(controlled),
            'reference': result_key(official),
            'comparison_class': 'prompt-ablation',
            **metric_deltas(controlled, official),
        })
    for budget in sorted({int(row['budget']) for row in search_rows if 'budget' in row}):
        uniform = keyed.get(f'uniform-one-shot-{budget}f')
        if not uniform:
            continue
        for candidate in search_rows:
            if int(candidate.get('budget', -1)) != budget or candidate is uniform:
                continue
            latency_ratio = safe_ratio(
                candidate.get('synchronized_model_seconds'),
                uniform.get('synchronized_model_seconds'),
            )
            name = (
                f'hierarchical-vs-uniform-at-{budget}-frames'
                if candidate.get('schedule') == 'embedding-window-local'
                else f'{candidate["schedule"]}-vs-uniform-at-{budget}-frames'
            )
            comparisons.append({
                'name': name,
                'candidate': result_key(candidate),
                'reference': result_key(uniform),
                'comparison_class': (
                    'latency-matched'
                    if latency_ratio is not None and abs(latency_ratio - 1.0) <= 0.05
                    else 'pareto'
                ),
                'synchronized_model_latency_ratio': latency_ratio,
                **metric_deltas(candidate, uniform),
            })
    if controlled:
        for row in search_rows:
            if row.get('schedule') == 'embedding-window-local':
                comparisons.append({
                    'name': f'{result_key(row)}-vs-paper-2fps-controlled',
                    'candidate': result_key(row),
                    'reference': result_key(controlled),
                    'comparison_class': 'accuracy-only-different-input-policy',
                    **metric_deltas(row, controlled),
                })
    return {
        'dataset': 'OMTGBench',
        'model': baseline_model or search_model,
        'primary_metric': 'EtF1',
        'results': all_rows,
        'comparisons': comparisons,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits: int = 2) -> str:
    if value is None or value == '':
        return '—'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, float)):
        return f'{float(value):.{digits}f}'
    return str(value)


def markdown_table(headers: list[tuple[str, str]], rows: list[dict]) -> list[str]:
    lines = [
        '| ' + ' | '.join(label for _, label in headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row.get(key)) for key, _ in headers) + ' |')
    return lines


def delta_with_ci(row: dict, metric: str) -> str:
    delta = float(row[f'delta_{metric}'])
    low = float(row[f'ci95_low_{metric}'])
    high = float(row[f'ci95_high_{metric}'])
    return f'{delta:+.2f} [{low:+.2f}, {high:+.2f}]'


def percent_reduction(ratio) -> str:
    if ratio is None:
        return '—'
    return f'{(1.0 - float(ratio)) * 100:.1f}%'


def comparison_map(result: dict) -> dict[str, dict]:
    return {row['name']: row for row in result['comparisons']}


def result_map(result: dict) -> dict[str, dict]:
    return {row['result_id']: row for row in result['results']}


def selected_comparison_rows(result: dict) -> list[dict]:
    selected = []
    for row in result['comparisons']:
        if row['comparison_class'] in {
            'prompt-ablation',
            'input-policy-tradeoff',
            'model-comparison',
        } or 'embedding-window-local' in row['name']:
            selected.append({
                'comparison': row['name'],
                **{metric: delta_with_ci(row, metric) for metric in REPORT_METRICS},
            })
    return selected


def write_markdown(path: Path, result: dict) -> None:
    rows = result['results']
    comparisons = comparison_map(result)
    indexed = result_map(result)
    audit = result['prompt_audit']
    tl64 = indexed.get('timelens2-4b/embedding-window-local-64f', {})
    tl128 = indexed.get('timelens2-4b/embedding-window-local-128f', {})
    qwen64 = indexed.get('qwen3-vl-4b/embedding-window-local-64f', {})
    qwen_uniform = indexed.get('qwen3-vl-4b/uniform-one-shot-64f', {})
    tl2fps = indexed.get('timelens2-4b/paper-2fps-controlled', {})
    qwen2fps = indexed.get('qwen3-vl-4b/paper-2fps-controlled', {})

    lines = [
        '# OMTG inference, prompting, and efficiency',
        '',
        '## Research assessment',
        '',
        'This report compiles all complete 320-query runs for TimeLens2-4B and '
        'Qwen3-VL-4B. Metric deltas use paired 95% percentile-bootstrap confidence '
        'intervals clustered by video '
        f'({result["bootstrap"]["samples"]:,} resamples; fixed seed '
        f'`{result["bootstrap"]["seed"]}`). Intervals are exploratory and are not '
        'corrected for multiple comparisons.',
        '',
        '| Proposed claim | Verdict | Fair interpretation |',
        '| --- | --- | --- |',
        '| Multi-span grounding is also an inference problem | **Supported, with scope limits** | '
        f'On base Qwen, embedding-window-local raises C-Acc from {fmt(qwen_uniform.get("C-Acc"))} '
        f'to {fmt(qwen64.get("C-Acc"))} and EtF1 from {fmt(qwen_uniform.get("EtF1"))} '
        f'to {fmt(qwen64.get("EtF1"))} without OMTG-specific training. This shows inference '
        'policy matters; it does not show training is unnecessary. |',
        '| Our method improves TimeLens2 over 2 FPS | **Partially supported** | '
        f'At 64 frames it improves cardinality but loses substantial localization accuracy. '
        f'At 128 frames it reaches C-Acc {fmt(tl128.get("C-Acc"))} and EtF1 '
        f'{fmt(tl128.get("EtF1"))}, while approximately matching the controlled 2 FPS '
        'thresholded localization scores. |',
        '| OMTG has a prompt flaw | **Supported as a benchmark confound** | '
        f'{audit.get("singular_instruction_samples", 0)}/{audit.get("samples", 0)} official '
        'instructions ask for a singular segment, while every label is multi-span. Call this '
        'an instruction–annotation mismatch, not evidence that the underlying videos or labels '
        'are flawed. |',
        '| Our method is cheaper | **Supported versus 2 FPS only** | Fixed-frame inference '
        'uses far fewer frames and less total time than 2 FPS. Against fixed-frame uniform '
        'inference, routing can reduce synchronized model latency yet increase wall time. |',
        '| Qwen improves beyond C-Acc | **Supported** | Embedding-64 improves EtF1, tIoU, '
        'tF1@0.3, and tF1@0.5 over controlled 2 FPS; tF1@0.7 is effectively tied. |',
        '',
        '## Complete results',
        '',
        'Accuracy values are percentages. Lower CardinalityError is better.',
    ]
    for model in result['models']:
        model_rows = [row for row in rows if row['model'] == model]
        lines.extend([
            '',
            f'### {model_label(model)}',
            '',
            *markdown_table([
                ('result', 'Setting'),
                ('C-Acc', 'C-Acc ↑'),
                ('CardinalityError', 'Card. error ↓'),
                ('EtF1', 'EtF1 ↑'),
                ('tIoU', 'tIoU ↑'),
                ('tF1@0.3', 'tF1@0.3 ↑'),
                ('tF1@0.5', 'tF1@0.5 ↑'),
                ('tF1@0.7', 'tF1@0.7 ↑'),
            ], model_rows),
        ])

    lines.extend([
        '',
        '## Key paired comparisons',
        '',
        'Each cell is `delta [95% CI]`; positive favors the candidate.',
        '',
        *markdown_table([
            ('comparison', 'Candidate vs reference'),
            ('C-Acc', 'Δ C-Acc'),
            ('EtF1', 'Δ EtF1'),
            ('tIoU', 'Δ tIoU'),
            ('tF1@0.3', 'Δ F1@0.3'),
            ('tF1@0.5', 'Δ F1@0.5'),
            ('tF1@0.7', 'Δ F1@0.7'),
        ], selected_comparison_rows(result)),
        '',
        '## Efficiency',
        '',
        *markdown_table([
            ('model_label', 'Model'),
            ('result', 'Setting'),
            ('input_frames', 'Input frames'),
            ('model_calls', 'Grounder calls'),
            ('synchronized_model_seconds', 'Sync. model latency (s)'),
            ('standalone_wall_seconds', 'Wall time (s)'),
            ('peak_vram_gib', 'Peak VRAM (GiB)'),
        ], rows),
        '',
        'Relative to controlled 2 FPS:',
        '',
    ])
    for name in (
        'timelens2-4b:embedding-64f-vs-controlled-2fps',
        'timelens2-4b:embedding-128f-vs-controlled-2fps',
        'qwen3-vl-4b:embedding-64f-vs-controlled-2fps',
    ):
        row = comparisons.get(name)
        if row:
            lines.append(
                f'- `{name}`: {percent_reduction(row["input_frame_ratio"])} fewer frames, '
                f'{percent_reduction(row["synchronized_model_latency_ratio"])} less '
                f'synchronized latency, and {percent_reduction(row["standalone_wall_ratio"])} '
                'less wall time.'
            )

    lines.extend([
        '',
        'The historical `gpu_seconds` field is synchronized end-to-end model-call '
        'latency, not pure CUDA kernel time. Runtime values are descriptive because '
        'each setting was executed once.',
        '',
        '## Prompt audit',
        '',
        f'- All {audit.get("singular_instruction_samples", 0)} of '
        f'{audit.get("samples", 0)} official prompts request “the video segment” and '
        '“its start and end seconds.”',
        f'- All {audit.get("multi_span_label_samples", 0)} labels contain multiple '
        f'spans: {audit.get("minimum_label_spans", 0)}–'
        f'{audit.get("maximum_label_spans", 0)}, mean '
        f'{fmt(audit.get("mean_label_spans"))}.',
        '- The controlled prompt changes three factors together: plural cardinality '
        'instruction, JSON output format, and an explicit duration bound. The current '
        'ablation cannot identify which component causes the gain.',
        '',
        *markdown_table([
            ('model', 'Model'),
            ('result', 'Prompt'),
            ('mean_predicted_spans', 'Mean predicted spans'),
            ('empty_outputs', 'Empty'),
            ('single_span_outputs', 'Single'),
            ('multi_span_outputs', 'Multiple'),
        ], [
            {**row, 'model': model_label(row['model'])}
            for row in audit.get('output_behavior', [])
        ]),
        '',
        '## Additional findings',
        '',
        '- **Naive decomposition hurts.** Full-video multipass is slower and less '
        'accurate than uniform one-shot for TimeLens2 at both budgets.',
        '- **Windowing drives cardinality; learned routing refines localization.** '
        'Uniform-window-local already captures most of the 64-frame C-Acc gain. At '
        '128 frames, embedding routing adds substantially more EtF1 and tIoU than '
        'cardinality accuracy.',
        '- **More frames do not monotonically improve counting.** TimeLens2 '
        'embedding-window-local gains localization quality from 64 to 128 frames, '
        'while C-Acc decreases slightly.',
        '- **Prompt sensitivity is model-dependent.** The controlled prompt improves '
        'all primary TimeLens2 metrics, but on Qwen it improves C-Acc and EtF1 while '
        'reducing tF1@0.3 and increasing cardinality error. Prompting is therefore a '
        'confound, not a universally effective substitute for model adaptation.',
        '- **Model specialization and inference are complementary in these runs.** '
        'Under the same embedding-64 '
        f'policy, TimeLens2 remains ahead of Qwen in C-Acc ({fmt(tl64.get("C-Acc"))} '
        f'vs {fmt(qwen64.get("C-Acc"))}) and EtF1 ({fmt(tl64.get("EtF1"))} vs '
        f'{fmt(qwen64.get("EtF1"))}). At controlled 2 FPS, the gap is also large '
        f'({fmt(tl2fps.get("EtF1"))} vs {fmt(qwen2fps.get("EtF1"))} EtF1).',
        '- Fixed-bin descriptive breakdowns by duration, span count, median span '
        'duration, evidence density, and temporal dispersion are available in '
        '`omtg_inference_strata.csv`; they are exploratory rather than separately '
        'powered hypothesis tests.',
        '',
        '## Limitations and reviewer concerns',
        '',
        '- The evidence covers one benchmark and two related 4B checkpoints. It is '
        'not yet sufficient for a general claim about all multi-span grounding.',
        '- TimeLens2-versus-Qwen is a checkpoint comparison, not a controlled training '
        'ablation. The models may differ in data, objectives, and implementation, so '
        'their gap cannot be attributed to OMTG training alone.',
        '- The controlled-prompt ablation is bundled; use separate plural-instruction, '
        'format, and duration ablations before claiming context engineering is more '
        'important than training.',
        '- The 64-frame embedding setting uses 20,628 aggregate frames instead of '
        '20,480, a 148-frame (0.72%) budget overflow. The 128-frame run has no overflow.',
        '- The 2 FPS frame count is duration-derived rather than decoder-measured.',
        '- Router work was executed once and reused across TimeLens2 budgets. Do not '
        'sum that cost twice when reconstructing the combined experiment.',
        '- Model loading is excluded, and one run per setting cannot establish '
        'runtime variance, energy use, or monetary cost.',
        '- Bootstrap intervals resample videos to preserve dependence among queries '
        'sharing a video; they do not quantify uncertainty over '
        'random seeds, prompt variants, hardware runs, or model sampling.',
        '',
        '## Novelty positioning',
        '',
        'The completed v1 is best presented as a controlled study of embedding-routed '
        'local inference for frozen set-valued grounders. Hierarchical and training-free '
        'selection are established in [DAFS](https://arxiv.org/abs/2607.15689), '
        '[SemVID](https://arxiv.org/abs/2603.05663), '
        '[TFVTG](https://arxiv.org/abs/2408.16219), and '
        '[CoMET-Agent](https://arxiv.org/abs/2606.15320); therefore '
        'the report does not claim the first training-free adaptive frame allocator. '
        'The proposed v2 contribution is narrower: deterministic residual set search '
        'with hard budget accounting and adaptive stopping.',
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        'json': output_dir / 'omtg_inference_comparison.json',
        'results_csv': output_dir / 'omtg_inference_results.csv',
        'deltas_csv': output_dir / 'omtg_inference_deltas.csv',
        'strata_csv': output_dir / 'omtg_inference_strata.csv',
        'markdown': output_dir / 'comparison.md',
    }


def write_outputs(output_dir: Path, result: dict) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    paths['json'].write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    write_csv(paths['results_csv'], result['results'])
    write_csv(paths['deltas_csv'], result['comparisons'])
    write_csv(paths['strata_csv'], result['stratified_results'])
    write_markdown(paths['markdown'], result)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-summary', type=Path, action='append', required=True)
    parser.add_argument('--search-summary', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument('--bootstrap-seed', type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_multi_model_comparison(
        args.baseline_summary,
        args.search_summary,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    paths = write_outputs(args.output_dir, result)
    print('model, result, EtF1, C-Acc, tIoU, synchronized_model_seconds')
    for row in result['results']:
        print(
            f'{row["model_label"]}, {row["result"]}, {row.get("EtF1", "")}, '
            f'{row.get("C-Acc", "")}, {row.get("tIoU", "")}, '
            f'{row.get("synchronized_model_seconds", "")}'
        )
    for path in paths.values():
        print(f'Wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
