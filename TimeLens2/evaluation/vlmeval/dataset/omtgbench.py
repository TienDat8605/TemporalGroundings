"""OMTG Bench dataset adapter and official one-to-many grounding metrics.

The metric implementation follows the evaluator released with OMTG Bench.  It
is kept here (rather than delegated to an LLM judge) so every experiment is
deterministic and can be resumed or rescored offline.
"""

from __future__ import annotations

import os
import re
import zipfile
from typing import Iterable

import numpy as np
from huggingface_hub import snapshot_download
from scipy.optimize import linear_sum_assignment

from ..smp import LMUDataRoot, dump, load
from .video_base import VideoBaseDataset


def parse_time_to_seconds(value: str) -> float:
    parts = value.strip().split(':')
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(f'Invalid time value: {value!r}')


def parse_time_intervals(text: str) -> list[list[float]]:
    """Parse the output formats accepted by the official OMTG evaluator."""
    intervals: list[list[float]] = []

    def add(start: str, end: str, converter) -> None:
        try:
            parsed = [converter(start.strip()), converter(end.strip())]
        except (TypeError, ValueError):
            return
        if parsed[1] > parsed[0]:
            intervals.append(parsed)

    patterns = [
        (r'<time>(\S+?)\s*-\s*(\S+?)\s*seconds?</time>', parse_time_to_seconds),
        (r'(\d+(?::\d+(?:\.\d+)?)?(?:\.\d+)?)\s*-\s*'
         r'(\d+(?::\d+(?:\.\d+)?)?(?:\.\d+)?)\s*seconds?', parse_time_to_seconds),
        (r'start\s*:\s*(\d+(?::\d+(?:\.\d+)?)?(?:\.\d+)?)\s*,\s*'
         r'end\s*:\s*(\d+(?::\d+(?:\.\d+)?)?(?:\.\d+)?)', parse_time_to_seconds),
        (r'starts\s+at\s+(\S+?)(?:\s+seconds?)?\s+and\s+ends\s+at\s+'
         r'(\S+?)(?:\s+seconds?)?', parse_time_to_seconds),
        (r'start\s+is\s+at\s+(\S+?)(?:\s+seconds?)?\s+and\s+(?:the\s+)?'
         r'end\s+is\s+at\s+(\S+?)(?:\s+seconds?)?', parse_time_to_seconds),
        (r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', float),
        (r'(?<!\d)(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?!\s*(?:seconds?|</time>))', float),
        (r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', float),
    ]
    # The official evaluator gives tagged/unit forms precedence, then accepts
    # all of its remaining free-form representations.
    for index, (pattern, converter) in enumerate(patterns):
        for start, end in re.findall(pattern, str(text), re.IGNORECASE):
            add(start, end, converter)
        if intervals and index < 2:
            break

    seen: set[tuple[float, float]] = set()
    unique: list[list[float]] = []
    for start, end in sorted(intervals):
        key = (start, end)
        if key not in seen:
            seen.add(key)
            unique.append([start, end])
    return unique


def temporal_iou(first: Iterable[float], second: Iterable[float]) -> float:
    start1, end1 = first
    start2, end2 = second
    intersection = max(0.0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0.0


def merge_overlapping_segments(segments: list[list[float]]) -> list[list[float]]:
    if not segments:
        return []
    merged = [list(item) for item in sorted(segments)]
    output = [merged[0]]
    for start, end in merged[1:]:
        if start <= output[-1][1]:
            output[-1][1] = max(output[-1][1], end)
        else:
            output.append([start, end])
    return output


def compute_one_to_many_metrics(
    predictions: list[list[float]],
    targets: list[list[float]],
    iou_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict[str, float]:
    """Compute OMTG C-Acc, union tIoU, Hungarian tF1, and EtF1."""
    cardinality_accuracy = float(len(predictions) == len(targets))
    pred_union = merge_overlapping_segments(predictions)
    target_union = merge_overlapping_segments(targets)
    pred_length = sum(end - start for start, end in pred_union)
    target_length = sum(end - start for start, end in target_union)
    intersection = sum(
        max(0.0, min(pred_end, target_end) - max(pred_start, target_start))
        for pred_start, pred_end in pred_union
        for target_start, target_end in target_union
    )
    union = pred_length + target_length - intersection
    results = {
        'C-Acc': cardinality_accuracy,
        'tIoU': intersection / union if union > 0 else 0.0,
        'CardinalityError': float(abs(len(predictions) - len(targets))),
    }

    if not predictions or not targets:
        value = float(not predictions and not targets)
        for threshold in iou_thresholds:
            results[f'tP@{threshold}'] = value
            results[f'tR@{threshold}'] = value
            results[f'tF1@{threshold}'] = value
        # This matches the official empty-set special case.
        results['EtF1'] = value
        return results

    matrix = np.asarray([
        [temporal_iou(prediction, target) for target in targets]
        for prediction in predictions
    ])
    pred_indices, target_indices = linear_sum_assignment(-matrix)
    matched_ious = [matrix[pred, target] for pred, target in zip(pred_indices, target_indices)]
    for threshold in iou_thresholds:
        true_positives = sum(value >= threshold for value in matched_ious)
        precision = true_positives / len(predictions)
        recall = true_positives / len(targets)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[f'tP@{threshold}'] = precision
        results[f'tR@{threshold}'] = recall
        results[f'tF1@{threshold}'] = f1
    results['EtF1'] = cardinality_accuracy * float(np.mean([
        results[f'tF1@{threshold}'] for threshold in iou_thresholds
    ]))
    return results


class OMTGBench(VideoBaseDataset):
    TYPE = 'Video-Temporal-Grounding'
    MODALITY = 'VIDEO'
    HF_REPO_ID = 'insomnia7/omtg_bench'

    @classmethod
    def supported_datasets(cls):
        return ['OMTGBench']

    def prepare_dataset(self, dataset_name='OMTGBench'):
        root = os.environ.get('OMTG_BENCH_ROOT')
        if not root:
            data_root = os.environ.get('TIMELENS2_DATA_ROOT')
            root = os.path.join(data_root, 'OMTGBench') if data_root else os.path.join(LMUDataRoot(), 'OMTGBench')
        data_file = os.path.join(root, 'OMTGBench.tsv')
        video_root = os.path.join(root, 'videos')
        if not os.path.isfile(data_file) or not os.path.isdir(video_root):
            snapshot_download(repo_id=self.HF_REPO_ID, repo_type='dataset', local_dir=root)
            archive = os.path.join(root, 'videos.zip')
            if os.path.isfile(archive) and not os.path.isdir(video_root):
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(root)
        if not os.path.isfile(data_file) or not os.path.isdir(video_root):
            raise FileNotFoundError(
                f'OMTG Bench is incomplete under {root}; run scripts/download_omtg_bench.sh'
            )
        return {'root': video_root, 'data_file': data_file}

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if video_llm:
            media = [{'type': 'video', 'value': os.path.join(self.data_root, line['video'])}]
        else:
            media = [{'type': 'image', 'value': path} for path in self.save_video_frames(line['video'])]
        return media + [{'type': 'text', 'value': line['question']}]

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        rows = [
            compute_one_to_many_metrics(
                parse_time_intervals(str(row['prediction'])),
                parse_time_intervals(str(row['answer'])),
            )
            for _, row in data.iterrows()
        ]
        results = {
            key: float(np.mean([row[key] for row in rows])) * (1.0 if key == 'CardinalityError' else 100.0)
            for key in rows[0]
        } if rows else {}
        score_file = os.path.splitext(eval_file)[0] + '_score.json'
        dump(results, score_file)
        return results
