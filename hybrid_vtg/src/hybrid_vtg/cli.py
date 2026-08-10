"""Command-line interface for one explicit benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import BENCHMARKS, METHODS, MODELS, load_builtin_plugins
from .runner import run_benchmark
from .sampling import SUPPORTED_PERCENTAGES


def parser() -> argparse.ArgumentParser:
    load_builtin_plugins()
    root = argparse.ArgumentParser(prog="hybrid-vtg")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one test-only benchmark/model/method setting")
    run.add_argument("--benchmark", choices=BENCHMARKS.names(), required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--model", choices=MODELS.names(), required=True)
    run.add_argument("--method", choices=METHODS.names(), required=True)
    run.add_argument("--subset", type=int, choices=SUPPORTED_PERCENTAGES, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--checkpoint", help="optional override; required for UniVTG")
    run.add_argument(
        "--model-spec",
        choices=("clip-b16", "clip-b32", "slowfast-clip-b32"),
        help="UniVTG raw feature stack when checkpoint metadata is insufficient",
    )
    run.add_argument(
        "--feature-root",
        action="append",
        type=Path,
        default=[],
        help="UniVTG directory containing <video-id>.npz; repeat to concatenate feature streams",
    )
    return root


def main() -> int:
    args = parser().parse_args()
    result = run_benchmark(
        benchmark_name=args.benchmark,
        data=args.data,
        model_name=args.model,
        method_name=args.method,
        percentage=args.subset,
        seed=args.seed,
        checkpoint=args.checkpoint,
        model_spec=args.model_spec,
        feature_roots=tuple(args.feature_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
