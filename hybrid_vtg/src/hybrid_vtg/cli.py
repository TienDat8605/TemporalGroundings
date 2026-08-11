"""Command-line interface for one explicit benchmark run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .downloads import TARGETS, download_assets
from .registry import BENCHMARKS, METHODS, MODELS, load_builtin_plugins
from .runner import run_benchmark


def parser() -> argparse.ArgumentParser:
    load_builtin_plugins()
    root = argparse.ArgumentParser(prog="hybrid-vtg")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one test-only benchmark/model/method setting")
    run.add_argument("--benchmark", choices=BENCHMARKS.names(), required=True)
    run.add_argument(
        "--data",
        type=Path,
        help="benchmark directory; defaults to ./assets/datasets/<benchmark>",
    )
    run.add_argument("--model", choices=MODELS.names(), required=True)
    run.add_argument("--method", choices=METHODS.names(), required=True)
    run.add_argument(
        "--subset",
        type=float,
        required=True,
        help="seeded query percentage from 0 through 100; decimals are accepted",
    )
    run.add_argument("--seed", type=int, required=True)
    run.add_argument(
        "--corridor-top-k",
        type=int,
        default=4,
        help="number of adaptive corridors retained by unitime-adaptive (1-8)",
    )
    run.add_argument(
        "--rerun",
        action="store_true",
        help="discard cached predictions and re-evaluate every selected sample",
    )
    run.add_argument(
        "--encoder-pruning",
        choices=("none", "mage"),
        default="none",
        help="vision-encoder pruning policy; mage uses motion/residual-guided complete cells",
    )
    run.add_argument(
        "--encoder-retention",
        type=float,
        default=1.0,
        help="fraction of dense Qwen merger cells retained inside the vision encoder",
    )
    run.add_argument(
        "--encoder-prune-layer",
        type=int,
        default=0,
        help="vision block index before which Mage-style selection is applied",
    )
    run.add_argument(
        "--post-pruning",
        choices=("none", "semvid"),
        default="none",
        help="post-encoder pruning policy applied immediately before model prediction",
    )
    run.add_argument(
        "--post-retention",
        type=float,
        default=1.0,
        help="final visual-token fraction relative to the original dense encoder output",
    )
    run.add_argument("--checkpoint", help="optional override; required for UniVTG")
    run.add_argument(
        "--base-checkpoint",
        help="optional Qwen2-VL base override for the UniTime adapter",
    )
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
    download = commands.add_parser("download", help="download datasets and checkpoints into one assets tree")
    download.add_argument(
        "targets",
        nargs="*",
        choices=TARGETS,
        help="assets to download; omit to download all datasets and checkpoints",
    )
    download.add_argument("--root", type=Path, default=Path("assets"), help="destination assets directory")
    download.add_argument(
        "--accept-licenses",
        action="store_true",
        help="confirm that you reviewed and accept every selected upstream license and dataset term",
    )
    download.add_argument(
        "--hf-login",
        action="store_true",
        help="log in to Hugging Face (uses HF_TOKEN when set, otherwise prompts securely)",
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "download":
        result = download_assets(
            args.root,
            args.targets,
            accept_licenses=args.accept_licenses,
            hf_login=args.hf_login,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    data = args.data or Path("assets") / "datasets" / args.benchmark
    result = run_benchmark(
        benchmark_name=args.benchmark,
        data=data,
        model_name=args.model,
        method_name=args.method,
        percentage=args.subset,
        seed=args.seed,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        model_spec=args.model_spec,
        feature_roots=tuple(args.feature_root),
        rerun=args.rerun,
        corridor_top_k=args.corridor_top_k,
        encoder_pruning=args.encoder_pruning,
        encoder_retention=args.encoder_retention,
        encoder_prune_layer=args.encoder_prune_layer,
        post_pruning=args.post_pruning,
        post_retention=args.post_retention,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
