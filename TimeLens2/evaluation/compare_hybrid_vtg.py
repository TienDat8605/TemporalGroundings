#!/usr/bin/env python3
"""Compare deterministic hybrid VTG runs with paired video-clustered bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ('mIoU', 'R@0.3', 'R@0.5', 'R@0.7')


def load_result(path: Path) -> dict:
    result = json.loads(path.read_text(encoding='utf-8'))
    if 'summary' not in result or 'records' not in result:
        raise ValueError(f'invalid hybrid summary: {path}')
    return result


def _seed(base: int, metric: str) -> int:
    digest = hashlib.sha256(metric.encode('utf-8')).digest()
    return base + int.from_bytes(digest[:4], 'little')


def cluster_bootstrap(
    candidate: dict[str, float],
    reference: dict[str, float],
    clusters: dict[str, str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    ids = sorted(set(candidate) & set(reference) & set(clusters))
    if not ids:
        raise ValueError('candidate and reference have no paired records')
    unique = sorted({clusters[row_id] for row_id in ids})
    members = {
        cluster: [row_id for row_id in ids if clusters[row_id] == cluster]
        for cluster in unique
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sampled = [row_id for cluster in selected for row_id in members[cluster]]
        deltas[index] = float(np.mean([
            candidate[row_id] - reference[row_id] for row_id in sampled
        ]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    observed = float(np.mean([candidate[row_id] - reference[row_id] for row_id in ids]))
    return {
        'samples': len(ids),
        'delta': observed,
        'ci95_low': float(low),
        'ci95_high': float(high),
        'p_value': min(1.0, 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))),
    }


def compare(candidate: dict, reference: dict, samples: int, seed: int) -> list[dict]:
    candidate_rows = {str(row['id']): row for row in candidate['records']}
    reference_rows = {str(row['id']): row for row in reference['records']}
    clusters = {str(row['id']): str(row['video']) for row in candidate['records']}
    output = []
    for metric in METRICS:
        stats = cluster_bootstrap(
            {row_id: float(row[metric]) for row_id, row in candidate_rows.items()},
            {row_id: float(row[metric]) for row_id, row in reference_rows.items()},
            clusters,
            samples=samples,
            seed=_seed(seed, metric),
        )
        scale = 100.0
        output.append({
            'metric': metric,
            **{key: value * scale if key in ('delta', 'ci95_low', 'ci95_high') else value
               for key, value in stats.items()},
        })
    return output


def write_report(path: Path, candidate: dict, reference: dict, comparisons: list[dict]) -> None:
    candidate_summary = candidate['summary']
    reference_summary = reference['summary']
    lines = [
        '# Hybrid VTG comparison',
        '',
        f'- Candidate: `{candidate_summary.get("dataset")}` / `{candidate_summary.get("backend")}`',
        f'- Reference: `{reference_summary.get("dataset")}` / `{reference_summary.get("backend")}`',
        '',
        '| Metric | Samples | Delta | 95% CI | p |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in comparisons:
        lines.append(
            f'| {row["metric"]} | {row["samples"]} | {row["delta"]:.2f} | '
            f'[{row["ci95_low"]:.2f}, {row["ci95_high"]:.2f}] | {row["p_value"]:.4f} |'
        )
    lines.extend([
        '',
        '## Efficiency',
        '',
        '| Run | Frames | Dense tokens | Retained tokens | Retention |',
        '|---|---:|---:|---:|---:|',
    ])
    for label, summary in (('Candidate', candidate_summary), ('Reference', reference_summary)):
        lines.append(
            f'| {label} | {summary.get("total_frames", summary.get("decoded_frames", 0))} | '
            f'{summary.get("dense_tokens", 0)} | {summary.get("retained_tokens", 0)} | '
            f'{summary.get("token_retention", 0):.4f} |'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=20260804)
    args = parser.parse_args()
    candidate = load_result(args.candidate)
    reference = load_result(args.reference)
    if candidate['summary'].get('dataset') != reference['summary'].get('dataset'):
        raise ValueError('candidate and reference datasets differ')
    comparisons = compare(candidate, reference, args.bootstrap_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'comparison.json').write_text(
        json.dumps({'comparisons': comparisons}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    pd.DataFrame(comparisons).to_csv(args.output_dir / 'comparison.csv', index=False)
    write_report(args.output_dir / 'comparison.md', candidate, reference, comparisons)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
