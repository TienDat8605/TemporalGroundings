#!/usr/bin/env python3
"""Compile per-benchmark VTG search summaries into one transfer report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORE_BENCHMARKS = (
    'vue-tr-v2',
    'momentseeker',
    'ego4d-nlq-v2',
    'qvhighlights-timelens',
)
LONG_BENCHMARKS = frozenset(CORE_BENCHMARKS[:3])


def load_summaries(run_root: Path, allow_incomplete: bool = False) -> dict[str, dict]:
    output = {}
    for benchmark in CORE_BENCHMARKS:
        path = run_root / benchmark / 'summary.json'
        if not path.is_file():
            if allow_incomplete:
                continue
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding='utf-8'))
        incomplete = [row for row in summary.get('results', []) if not row.get('complete')]
        if incomplete and not allow_incomplete:
            raise ValueError(f'{benchmark} has {len(incomplete)} incomplete settings')
        output[benchmark] = summary
    return output


def benchmark_verdict(summary: dict) -> dict:
    comparisons = summary.get('comparisons', [])
    positive = any(row.get('significant_positive') for row in comparisons)
    negative = any(row.get('significant_negative') for row in comparisons)
    return {
        'supports_transfer': positive and not negative,
        'has_significant_gain': positive,
        'has_significant_harm': negative,
    }


def compile_result(summaries: dict[str, dict]) -> dict:
    verdicts = {
        benchmark: benchmark_verdict(summary) for benchmark, summary in summaries.items()
    }
    long_support = sum(
        verdicts.get(benchmark, {}).get('supports_transfer', False)
        for benchmark in LONG_BENCHMARKS
    )
    long_form_complete = LONG_BENCHMARKS.issubset(summaries)
    return {
        'benchmarks': list(summaries),
        'verdicts': verdicts,
        'long_form_support_count': long_support,
        'long_form_complete': long_form_complete,
        'broad_long_form_transfer_supported': long_support >= 2 if long_form_complete else None,
    }


def flattened_rows(summaries: dict[str, dict], field: str) -> list[dict]:
    return [
        {'dataset': benchmark, **row}
        for benchmark, summary in summaries.items()
        for row in summary.get(field, [])
    ]


def write_markdown(path: Path, summaries: dict[str, dict], result: dict) -> None:
    lines = [
        '# Cross-benchmark hierarchical test-time search',
        '',
        'The confirmatory comparison is embedding-window-local versus uniform-one-shot '
        'under the controlled prompt. Budgets 64 and 128 are co-primary with Holm '
        'correction within each benchmark.',
        '',
        '## Transfer verdict',
        '',
        '| Benchmark | Significant gain | Significant harm | Supports transfer |',
        '| --- | :---: | :---: | :---: |',
    ]
    for benchmark in CORE_BENCHMARKS:
        if benchmark not in result['verdicts']:
            continue
        row = result['verdicts'][benchmark]
        lines.append(
            f'| {benchmark} | {"yes" if row["has_significant_gain"] else "no"} | '
            f'{"yes" if row["has_significant_harm"] else "no"} | '
            f'{"yes" if row["supports_transfer"] else "no"} |'
        )
    broad_verdict = result['broad_long_form_transfer_supported']
    broad_text = (
        'supported' if broad_verdict else 'not supported'
    ) if broad_verdict is not None else 'not evaluated (incomplete long-form suite)'
    lines.extend([
        '',
        f'Long-form benchmarks supporting transfer: **{result["long_form_support_count"]}/3**.',
        '',
        f'Broad long-form transfer is **{broad_text}** under the preregistered '
        'two-of-three rule.',
        '',
        '## Co-primary deltas',
        '',
        '| Benchmark | Budget | Δ mIoU | 95% CI | Holm p |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for benchmark, summary in summaries.items():
        for row in summary.get('comparisons', []):
            lines.append(
                f'| {benchmark} | {row["budget"]} | {row["delta"]:.2f} | '
                f'[{row["ci95_low"]:.2f}, {row["ci95_high"]:.2f}] | '
                f'{row["holm_p_value"]:.4f} |'
            )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_pareto_plots(output_dir: Path, summaries: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for benchmark, summary in summaries.items():
        rows = [
            row for row in summary.get('results', [])
            if row.get('complete') and row.get('prompt_mode') == 'controlled'
        ]
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for schedule in sorted({row['schedule'] for row in rows}):
            selected = [row for row in rows if row['schedule'] == schedule]
            selected.sort(key=lambda row: row.get('model_seconds', 0))
            axis.plot(
                [row.get('model_seconds', 0) for row in selected],
                [row.get('mIoU', 0) for row in selected],
                marker='o',
                label=schedule,
            )
            for row in selected:
                axis.annotate(
                    str(row['budget']),
                    (row.get('model_seconds', 0), row.get('mIoU', 0)),
                    xytext=(4, 4),
                    textcoords='offset points',
                    fontsize=8,
                )
        axis.set_title(benchmark)
        axis.set_xlabel('Synchronized model-call latency (s)')
        axis.set_ylabel('mIoU (%)')
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f'{benchmark}_pareto.png', dpi=160)
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summaries = load_summaries(args.run_root, allow_incomplete=args.allow_incomplete)
    result = compile_result(summaries)
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / 'combined_summary.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    pd.DataFrame(flattened_rows(summaries, 'results')).to_csv(
        args.run_root / 'combined_results.csv', index=False
    )
    pd.DataFrame(flattened_rows(summaries, 'comparisons')).to_csv(
        args.run_root / 'combined_comparisons.csv', index=False
    )
    write_markdown(args.run_root / 'combined_report.md', summaries, result)
    if not args.no_plots:
        write_pareto_plots(args.run_root, summaries)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
