"""Command-line interface for local GPU evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from .benchmarks import load_benchmark
from .config import SPATIAL_POLICIES, GrounderConfig, PipelineConfig, SpatialAllocatorConfig
from .doctor import inspect_runtime
from .io import append_jsonl, completed_ids, ensure_manifest, git_revision, read_jsonl
from .metrics import evaluate
from .optimization_validation import compare_optimization_runs
from .pipeline import HybridVTGPipeline
from .semvid_bridge import default_semvid_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybrid-vtg")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="validate the local runtime without loading model weights")
    doctor.add_argument("--semvid-root", type=Path, default=default_semvid_root())

    run = commands.add_parser("run", help="run full-video spatial-token grounding")
    run.add_argument(
        "--benchmark",
        choices=("omtg", "tacos", "charades-sta", "activitynet-grounding", "activitynet-captions", "jsonl"),
        required=True,
    )
    run.add_argument("--data", type=Path, required=True, help="dataset root, or a canonical JSONL for --benchmark jsonl")
    run.add_argument("--split", default=None)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--semvid-root", type=Path, default=default_semvid_root())
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--spatial-policy", choices=SPATIAL_POLICIES, default="semvid")
    run.add_argument("--model", default=GrounderConfig.model)
    run.add_argument("--expert-fps", type=float, default=GrounderConfig.fps)
    run.add_argument("--retention-ratio", type=float, default=SpatialAllocatorConfig.retention_ratio)
    run.add_argument("--max-new-tokens", type=int, default=GrounderConfig.max_new_tokens)
    run.add_argument(
        "--allow-thinking", action="store_true",
        help="allow Qwen3-VL-Thinking to reason before answering (slower and less format-stable)",
    )
    run.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=GrounderConfig.dtype)
    run.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default=GrounderConfig.attention)
    run.add_argument(
        "--optimization-profile", choices=("safe", "optimized"), default="safe",
        help="safe preserves serial inference; optimized enables batch-two and CPU prefetch",
    )
    run.add_argument("--qwen-batch-size", type=int, choices=(1, 2), default=None)
    run.add_argument("--qwen-pairing-lookahead", type=int, default=GrounderConfig.pairing_lookahead)
    run.add_argument("--preprocess-workers", type=int, choices=(0, 1), default=None)
    run.add_argument("--prefetch-depth", type=int, default=None)
    run.add_argument(
        "--capture-validation-logits", action="store_true",
        help="record first-token top logits for batch-equivalence checks (validation only)",
    )
    run.add_argument("--fail-fast", action="store_true")

    score = commands.add_parser("evaluate", help="compute standard VTG metrics from a result JSONL")
    score.add_argument("--input", type=Path, required=True)
    validate = commands.add_parser("validate-optimization", help="check batch/prefetch equivalence and speed")
    validate.add_argument("--baseline", type=Path, required=True)
    validate.add_argument("--candidate", type=Path, required=True)
    validate.add_argument("--minimum-samples", type=int, default=32)
    validate.add_argument("--minimum-speedup", type=float, default=0.0)
    validate.add_argument("--logit-tolerance", type=float, default=0.05)
    return parser


def _config(args: argparse.Namespace) -> PipelineConfig:
    optimized = args.optimization_profile == "optimized"
    qwen_batch_size = args.qwen_batch_size if args.qwen_batch_size is not None else (2 if optimized else 1)
    preprocess_workers = args.preprocess_workers if args.preprocess_workers is not None else (1 if optimized else 0)
    prefetch_depth = args.prefetch_depth if args.prefetch_depth is not None else (
        2 if optimized and preprocess_workers == 1 else 0
    )
    grounder = replace(
        GrounderConfig(), model=args.model, fps=args.expert_fps,
        max_new_tokens=args.max_new_tokens, force_stop_thinking=not args.allow_thinking,
        dtype=args.dtype, attention=args.attention,
        batch_size=qwen_batch_size, pairing_lookahead=args.qwen_pairing_lookahead,
        preprocess_workers=preprocess_workers, prefetch_depth=prefetch_depth,
        capture_validation_logits=args.capture_validation_logits,
    )
    spatial_allocator = replace(
        SpatialAllocatorConfig(), spatial_policy=args.spatial_policy,
        retention_ratio=args.retention_ratio,
    )
    return PipelineConfig(grounder=grounder, spatial_allocator=spatial_allocator)


def _run(args: argparse.Namespace) -> int:
    split = args.split or (
        "test" if args.benchmark in {"omtg", "charades-sta", "tacos"}
        else "val_2" if args.benchmark.startswith("activitynet") else "all"
    )
    samples = load_benchmark(args.benchmark, args.data, split, args.limit)
    config = _config(args)
    repository_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema": 5,
        "benchmark": args.benchmark,
        "data": str(args.data.resolve()),
        "split": split,
        "config": config.to_dict(),
        "project_revision": git_revision(repository_root),
        "hybrid_vtg_revision": git_revision(repository_root),
        "spatial_policy": config.spatial_allocator.spatial_policy,
        "allocator_constants": config.spatial_allocator.allocator_constants(),
        "semvid_root": str(args.semvid_root.resolve()),
        "semvid_revision": git_revision(args.semvid_root),
    }
    ensure_manifest(args.output.with_suffix(".manifest.json"), manifest)
    existing = read_jsonl(args.output)
    done = completed_ids(existing)
    pending = [sample for sample in samples if sample.id not in done]
    if not pending:
        metrics = evaluate(existing)
        metrics_path = args.output.with_suffix(".metrics.json")
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in (
                "processed_this_invocation", "inference_wall_seconds",
                "seconds_per_processed_sample", "samples_per_second",
            ):
                if key in previous:
                    metrics[key] = previous[key]
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    pipeline = HybridVTGPipeline(config, args.semvid_root)
    inference_started = perf_counter()
    processed = 0
    for sample, record, error in pipeline.iter_results(pending):
        if error is not None:
            if args.fail_fast:
                raise error
            record = {
                "id": sample.id,
                "video": sample.video,
                "query": sample.query,
                "duration": sample.duration,
                "targets": [list(value) for value in sample.targets],
                "group": sample.group,
                "cardinality": sample.cardinality,
                "spatial_policy": config.spatial_allocator.spatial_policy,
                "prediction": None,
                "error": repr(error),
            }
        assert record is not None
        append_jsonl(args.output, record)
        processed += 1
    inference_wall_seconds = perf_counter() - inference_started
    records = read_jsonl(args.output)
    metrics = evaluate(records)
    metrics["processed_this_invocation"] = processed
    metrics["inference_wall_seconds"] = inference_wall_seconds
    metrics["seconds_per_processed_sample"] = inference_wall_seconds / processed if processed else 0.0
    metrics["samples_per_second"] = processed / inference_wall_seconds if inference_wall_seconds else 0.0
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
    if args.command == "validate-optimization":
        def throughput(path: Path) -> float:
            metrics_path = path.with_suffix(".metrics.json")
            if not metrics_path.is_file():
                return 0.0
            return float(json.loads(metrics_path.read_text(encoding="utf-8")).get("samples_per_second", 0.0))

        result = compare_optimization_runs(
            read_jsonl(args.baseline), read_jsonl(args.candidate),
            minimum_samples=args.minimum_samples, minimum_speedup=args.minimum_speedup,
            logit_tolerance=args.logit_tolerance,
            baseline_throughput=throughput(args.baseline),
            candidate_throughput=throughput(args.candidate),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
