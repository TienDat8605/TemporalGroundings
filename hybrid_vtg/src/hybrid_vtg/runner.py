"""Single-run, resumable benchmark orchestration."""

from __future__ import annotations

import platform
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

from .contracts import Prediction, Sample
from .io import append_jsonl, ensure_manifest, git_revision, read_jsonl, write_json
from .registry import BENCHMARKS, METHODS, MODELS, load_builtin_plugins
from .results import refresh_results_index, run_directory
from .sampling import SAMPLER_SCHEMA, ordered_samples, percentage_key, subset_samples, validate_percentage

SCHEMA_VERSION = 1


def _pruning_variant(
    model: str,
    encoder_pruning: str,
    encoder_retention: float,
    encoder_prune_layer: int,
    post_pruning: str,
    post_retention: float,
) -> str:
    parts = [model]
    if encoder_pruning != "none":
        parts.append(f"enc-{encoder_pruning}-r{encoder_retention:g}-l{encoder_prune_layer}")
    if post_pruning != "none":
        parts.append(f"post-{post_pruning}-r{post_retention:g}")
    return "--".join(parts)


def _validate_pruning_configuration(
    model: str,
    encoder_pruning: str,
    encoder_retention: float,
    encoder_prune_layer: int,
    post_pruning: str,
    post_retention: float,
) -> None:
    if encoder_pruning not in {"none", "mage"} or post_pruning not in {"none", "semvid"}:
        raise ValueError("unknown pruning policy")
    if not 0 < encoder_retention <= 1 or not 0 < post_retention <= 1:
        raise ValueError("pruning retention ratios must be in (0, 1]")
    if encoder_prune_layer < 0:
        raise ValueError("encoder prune layer must be non-negative")
    if encoder_pruning == "none" and encoder_retention != 1.0:
        raise ValueError("encoder retention requires --encoder-pruning mage")
    if post_pruning == "none" and post_retention != 1.0:
        raise ValueError("post retention requires --post-pruning semvid")
    if encoder_pruning == "mage" and post_pruning == "semvid" and post_retention > encoder_retention:
        raise ValueError("post retention cannot exceed encoder retention when both policies are enabled")
    if model == "univtg" and (encoder_pruning != "none" or post_pruning != "none"):
        raise ValueError("Mage and SemVID pruning are available only for Qwen-based models")


def _record(sample: Sample, prediction: Prediction, seconds: float) -> dict[str, Any]:
    return {
        "id": sample.id,
        "video": sample.video,
        "query": sample.query,
        "duration": sample.duration,
        "targets": [list(value) for value in sample.targets],
        "group": sample.group,
        "cardinality": sample.cardinality,
        "prediction": prediction.to_dict(),
        "wall_seconds": seconds,
    }


def _evaluation_records(samples: list[Sample], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = {str(record["id"]): record for record in records}
    output = []
    for sample in samples:
        output.append(
            completed.get(
                sample.id,
                {
                    "id": sample.id,
                    "video": sample.video,
                    "query": sample.query,
                    "duration": sample.duration,
                    "targets": [list(value) for value in sample.targets],
                    "group": sample.group,
                    "cardinality": sample.cardinality,
                    "prediction": {"spans": [], "raw_output": "", "telemetry": {}},
                },
            )
        )
    return output


def _evaluation_summary(benchmark: Any, samples: list[Sample], records: list[dict[str, Any]]) -> dict[str, Any]:
    successful_ids = {str(record["id"]) for record in records}
    successful = sum(sample.id in successful_ids for sample in samples)
    failed = len(samples) - successful
    if not samples:
        return {
            "status": "empty",
            "requested": 0,
            "successful": 0,
            "failed": 0,
            "metrics": None,
            "message": "The requested subset contains no samples.",
        }
    if successful == 0:
        return {
            "status": "failed",
            "requested": len(samples),
            "successful": 0,
            "failed": failed,
            "metrics": None,
            "message": "No predictions succeeded; metrics were not calculated. See errors.jsonl.",
        }
    metrics = benchmark.evaluate(_evaluation_records(samples, records))
    if metrics is None:
        return {
            "status": "complete" if failed == 0 else "partial",
            "requested": len(samples),
            "successful": successful,
            "failed": failed,
            "metrics": None,
            "message": (
                "Predictions succeeded but this benchmark has no local ground truth "
                "(hidden-label split); metrics are unavailable. A submission file was exported."
            ),
        }
    return {
        "status": "complete" if failed == 0 else "partial",
        "requested": len(samples),
        "successful": successful,
        "failed": failed,
        "metrics": metrics,
        "metrics_scope": "all requested samples; failed samples count as empty predictions",
    }


def run_benchmark(
    *,
    benchmark_name: str,
    data: Path,
    model_name: str,
    method_name: str,
    percentage: float,
    seed: int,
    checkpoint: str | None = None,
    base_checkpoint: str | None = None,
    model_spec: str | None = None,
    feature_roots: tuple[Path, ...] = (),
    rerun: bool = False,
    encoder_pruning: str = "none",
    encoder_retention: float = 1.0,
    encoder_prune_layer: int = 0,
    post_pruning: str = "none",
    post_retention: float = 1.0,
) -> dict[str, Any]:
    percentage = validate_percentage(percentage)
    if base_checkpoint is not None and model_name != "unitime":
        raise ValueError("base checkpoint override is available only for UniTime")
    native_models = {"unitime", "timelens2-4b", "timelens-8b", "timelens-7b"}
    if method_name == "native" and model_name not in native_models:
        raise ValueError("native requires --model unitime, timelens2-4b, timelens-8b, or timelens-7b")
    if (
        method_name == "native"
        and model_name != "unitime"
        and (encoder_pruning != "none" or post_pruning != "none")
    ):
        raise ValueError(
            "native TimeLens inference is dense; use coarse-to-fine-64 for Mage or SemVID"
        )
    _validate_pruning_configuration(
        model_name,
        encoder_pruning,
        encoder_retention,
        encoder_prune_layer,
        post_pruning,
        post_retention,
    )
    if method_name in {"anchored-corridor-64", "sgde-64"} and (
        encoder_pruning != "none" or post_pruning != "none"
    ):
        raise ValueError(
            f"{method_name} currently requires dense evidence; run Mage and SemVID "
            "as separate follow-up ablations"
        )
    load_builtin_plugins()
    benchmark = BENCHMARKS.create(benchmark_name)
    data = data.expanduser().resolve()
    if not data.is_dir():
        default = (Path.cwd() / "assets" / "datasets" / benchmark_name).resolve()
        raise FileNotFoundError(
            f"benchmark data directory does not exist: {data}. "
            f"If you used the default downloader location, pass --data {default}"
        )
    samples = benchmark.load_test(data)
    ordered = ordered_samples(samples, seed)
    selected = subset_samples(samples, percentage, seed)

    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / "results"
    output_model = _pruning_variant(
        model_name,
        encoder_pruning,
        encoder_retention,
        encoder_prune_layer,
        post_pruning,
        post_retention,
    )
    output_method = method_name
    manifest = {
        "schema": SCHEMA_VERSION,
        "benchmark": benchmark_name,
        "split": "test",
        "data": str(data),
        "model": model_name,
        "checkpoint": checkpoint,
        "model_spec": model_spec,
        "feature_roots": [str(path.resolve()) for path in feature_roots],
        "method": method_name,
        "seed": seed,
        "sampler": SAMPLER_SCHEMA,
        "ordered_sample_ids": [sample.id for sample in ordered],
        "project_revision": git_revision(project_root.parent),
        "python": platform.python_version(),
        "batch_size": 1,
        "training_free": model_name != "unitime",
    }
    if model_name == "unitime":
        manifest["checkpoint"] = checkpoint or "zeqianli/UniTime"
        manifest["base_checkpoint"] = base_checkpoint or "Qwen/Qwen2-VL-7B-Instruct"
        manifest["post_hoc_training_free"] = True
        manifest["upstream_trained_adapter"] = True
        manifest["maximum_evidence_units"] = 4_096
    elif model_name == "qwen2-vl-7b":
        manifest["checkpoint"] = checkpoint or "Qwen/Qwen2-VL-7B-Instruct"
        manifest["maximum_evidence_units"] = 4_096
    elif model_name == "qwen3-vl-4b":
        manifest["checkpoint"] = checkpoint or "Qwen/Qwen3-VL-4B-Instruct"
        manifest["maximum_evidence_units"] = 4_096
    elif model_name == "timelens2-4b":
        manifest["checkpoint"] = checkpoint or "MCG-NJU/TimeLens2-4B"
        if method_name == "native":
            manifest["native_video_fps"] = 2.0
            manifest["native_total_pixel_budget"] = 4_096 * 32 * 32
        manifest["maximum_evidence_units"] = 4_096
    elif model_name == "timelens-8b":
        manifest["checkpoint"] = checkpoint or "TencentARC/TimeLens-8B"
        if method_name == "native":
            manifest["native_video_fps"] = 2.0
            manifest["native_total_pixel_budget"] = 4_096 * 32 * 32
        manifest["maximum_evidence_units"] = 4_096
    elif model_name == "timelens-7b":
        manifest["checkpoint"] = checkpoint or "TencentARC/TimeLens-7B"
        if method_name == "native":
            manifest["native_video_fps"] = 2.0
            manifest["native_total_pixel_budget"] = 4_096 * 28 * 28
        manifest["maximum_evidence_units"] = 4_096
    if output_model != model_name:
        manifest["result_model"] = output_model
        manifest["pruning"] = {
            "encoder_policy": encoder_pruning,
            "encoder_retention": encoder_retention,
            "encoder_layer": encoder_prune_layer,
            "post_policy": post_pruning,
            "post_retention": post_retention,
        }
    if method_name == "anchored-corridor-64":
        manifest["method_config"] = {
            "grounding_frame_budget": 64,
            "maximum_routed_windows": 16,
            "router_frames_per_window": 4,
            "query_views": ["raw", "coarse", "actions", "details"],
            "raw_query_weight": 0.5,
            "expanded_query_max_weight": 0.5,
            "routing_margin": 0.5,
            "corridor_context_seconds": 4.0,
            "corridor_max_seconds": 64.0,
            "maximum_encoder_calls": 1,
            "maximum_primary_grounder_calls": 1,
        }
    elif method_name == "sgde-64":
        manifest["method_config"] = {
            "grounding_frame_budget": 64,
            "scout_fps": 1.0,
            "context_seconds": 4.0,
            "num_anchors": 6,
            "maximum_encoder_calls": 1,
            "maximum_primary_grounder_calls": 1,
        }
    hyperparameters = {
        "benchmark": benchmark_name,
        "model": model_name,
        "checkpoint": manifest.get("checkpoint"),
        "base_checkpoint": manifest.get("base_checkpoint"),
        "model_spec": model_spec,
        "feature_roots": manifest["feature_roots"],
        "method": method_name,
        "seed": seed,
        "batch_size": manifest["batch_size"],
        "encoder_pruning": encoder_pruning,
        "encoder_retention": encoder_retention,
        "encoder_prune_layer": encoder_prune_layer,
        "post_pruning": post_pruning,
        "post_retention": post_retention,
        "method_config": manifest.get("method_config"),
    }
    run_dir = run_directory(
        results_root,
        benchmark_name,
        model_name,
        output_method,
        seed,
        hyperparameters=hyperparameters,
    )
    print(f"run directory: {run_dir}", flush=True)
    # A rerun intentionally replaces results produced by an older revision or
    # configuration. Normal resume mode still rejects incompatible manifests.
    ensure_manifest(run_dir / "manifest.json", manifest, replace=rerun)

    predictions_path = run_dir / "predictions.jsonl"
    if rerun:
        # Discard prior predictions so every selected sample is re-evaluated with the
        # current code, instead of resuming from a cached run.
        predictions_path.unlink(missing_ok=True)
        (run_dir / "errors.jsonl").unlink(missing_ok=True)
    existing = read_jsonl(predictions_path)
    done = {str(record["id"]) for record in existing}
    pending = [sample for sample in selected if sample.id not in done]
    if pending:
        method_options = {}
        if method_name == "sgde-64" and feature_roots:
            method_options["feature_roots"] = feature_roots
        method = METHODS.create(method_name, **method_options)
        method_cache = results_root / "cache" / "methods" / method_name
        # Batch CPU-only preprocessing (e.g. scene detection) runs before the model
        # backend is loaded, so it never competes with GPU memory.
        method.prepare(pending, method_cache)
        model_options = dict(
            cache_dir=results_root / "cache",
            checkpoint=checkpoint,
            model_spec=model_spec,
            feature_roots=feature_roots,
            encoder_pruning=encoder_pruning,
            encoder_retention=encoder_retention,
            encoder_prune_layer=encoder_prune_layer,
            post_pruning=post_pruning,
            post_retention=post_retention,
        )
        if model_name == "unitime":
            model_options["base_checkpoint"] = base_checkpoint
        model = MODELS.create(model_name, **model_options)
        method.validate_model(model)
        progress = tqdm(pending, desc=f"{benchmark_name}/{model_name}/{output_method}", unit="sample")
        for sample in progress:
            started = perf_counter()
            try:
                prediction = method.run(sample, model, method_cache)
            except Exception as error:
                append_jsonl(
                    run_dir / "errors.jsonl",
                    {
                        "id": sample.id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "wall_seconds": perf_counter() - started,
                    },
                )
                progress.set_postfix(failed="yes")
                continue
            append_jsonl(predictions_path, _record(sample, prediction, perf_counter() - started))
            progress.set_postfix(failed="no")

    records = read_jsonl(predictions_path)
    values = _evaluation_records(selected, records)
    summary = _evaluation_summary(benchmark, selected, records)
    summary["run_directory"] = str(run_dir)
    summary["hyperparameters"] = {**hyperparameters, "subset_percentage": percentage}
    metrics_name = f"metrics-{percentage_key(percentage)}.json"
    write_json(run_dir / metrics_name, summary)
    # Export a submission whenever the benchmark has no local ground truth (hidden-label
    # splits like QVHighlights) or the full test set is evaluated. For hidden-label
    # benchmarks the submission is the only result, so it is produced at any subset.
    if (summary["metrics"] is None or percentage == 100) and summary["failed"] == 0:
        submission = benchmark.export_submission(
            values,
            run_dir / f"submission-{percentage_key(percentage)}.jsonl",
        )
        if submission is not None:
            summary["submission"] = str(submission)
            write_json(run_dir / metrics_name, summary)
    refresh_results_index(results_root)
    return summary
