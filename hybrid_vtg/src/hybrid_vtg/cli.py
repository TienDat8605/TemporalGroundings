"""Command-line interface for local GPU evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .benchmarks import load_benchmark
from .config import CoarseConfig, PipelineConfig, RefinementConfig, SemVIDConfig
from .doctor import inspect_runtime
from .io import append_jsonl, completed_ids, ensure_manifest, git_revision, read_jsonl
from .metrics import evaluate
from .pipeline import HybridVTGPipeline
from .semvid_bridge import default_semvid_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybrid-vtg")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="validate the local runtime without loading model weights")
    doctor.add_argument("--semvid-root", type=Path, default=default_semvid_root())

    run = commands.add_parser("run", help="run hierarchical grounding")
    run.add_argument("--benchmark", choices=("charades-sta", "activitynet-grounding", "activitynet-captions", "jsonl"), required=True)
    run.add_argument("--data", type=Path, required=True, help="dataset root, or a canonical JSONL for --benchmark jsonl")
    run.add_argument("--split", default=None)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path, default=Path(".cache/hybrid-vtg"))
    run.add_argument("--semvid-root", type=Path, default=default_semvid_root())
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--coarse-model", default=CoarseConfig.checkpoint)
    run.add_argument("--coarse-fps", type=float, default=CoarseConfig.fps)
    run.add_argument("--coarse-max-frames", type=int, default=CoarseConfig.max_frames)
    run.add_argument("--temporal-budget", type=float, default=CoarseConfig.union_budget_seconds)
    run.add_argument("--no-temporal-prune", action="store_true", help="ablation: send the whole video to Qwen")
    run.add_argument("--semvid-model", default=SemVIDConfig.model)
    run.add_argument("--expert-fps", type=float, default=SemVIDConfig.fps)
    run.add_argument("--retention-ratio", type=float, default=SemVIDConfig.retention_ratio)
    run.add_argument("--no-spatial-prune", action="store_true", help="ablation: disable SemVID token pruning")
    run.add_argument("--max-new-tokens", type=int, default=SemVIDConfig.max_new_tokens)
    run.add_argument(
        "--allow-thinking", action="store_true",
        help="allow Qwen3-VL-Thinking to reason before answering (slower and less format-stable)",
    )
    run.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=SemVIDConfig.dtype)
    run.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default=SemVIDConfig.attention)
    run.add_argument("--timestamp-mode", choices=("absolute", "relative", "auto"), default=SemVIDConfig.timestamp_mode)
    run.add_argument("--no-refine", action="store_true")
    run.add_argument("--refine-fps", type=float, default=RefinementConfig.fps)
    run.add_argument("--fail-fast", action="store_true")

    score = commands.add_parser("evaluate", help="compute standard VTG metrics from a result JSONL")
    score.add_argument("--input", type=Path, required=True)
    return parser


def _config(args: argparse.Namespace) -> PipelineConfig:
    coarse = replace(
        CoarseConfig(), enabled=not args.no_temporal_prune,
        checkpoint=args.coarse_model, fps=args.coarse_fps,
        max_frames=args.coarse_max_frames, union_budget_seconds=args.temporal_budget,
    )
    semvid = replace(
        SemVIDConfig(), enabled=not args.no_spatial_prune,
        model=args.semvid_model, fps=args.expert_fps,
        retention_ratio=args.retention_ratio,
        max_new_tokens=args.max_new_tokens, force_stop_thinking=not args.allow_thinking,
        dtype=args.dtype, attention=args.attention, timestamp_mode=args.timestamp_mode,
    )
    refinement = replace(RefinementConfig(), enabled=not args.no_refine, fps=args.refine_fps)
    return PipelineConfig(coarse=coarse, semvid=semvid, refinement=refinement)


def _run(args: argparse.Namespace) -> int:
    split = args.split or ("test" if args.benchmark == "charades-sta" else "val_2")
    samples = load_benchmark(args.benchmark, args.data, split, args.limit)
    config = _config(args)
    manifest = {
        "schema": 1,
        "benchmark": args.benchmark,
        "data": str(args.data.resolve()),
        "split": split,
        "config": config.to_dict(),
        "semvid_root": str(args.semvid_root.resolve()),
        "semvid_revision": git_revision(args.semvid_root),
    }
    ensure_manifest(args.output.with_suffix(".manifest.json"), manifest)
    existing = read_jsonl(args.output)
    done = completed_ids(existing)
    pending = [sample for sample in samples if sample.id not in done]
    if not pending:
        metrics = evaluate(existing)
        args.output.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    pipeline = HybridVTGPipeline(config, args.cache_dir, args.semvid_root)
    for sample in pending:
        try:
            record = pipeline.run_sample(sample)
        except Exception as error:
            if args.fail_fast:
                raise
            record = {"id": sample.id, "video": sample.video, "query": sample.query, "error": repr(error)}
        append_jsonl(args.output, record)
    records = read_jsonl(args.output)
    metrics = evaluate(records)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "doctor":
        checks = inspect_runtime(args.semvid_root)
        for name, passed, detail in checks:
            print(f"[{'OK' if passed else 'FAIL'}] {name}: {detail}")
        return 0 if all(passed for _, passed, _ in checks) else 1
    if args.command == "evaluate":
        print(json.dumps(evaluate(read_jsonl(args.input)), indent=2, sort_keys=True))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
