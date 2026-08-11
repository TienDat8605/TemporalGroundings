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
    model_spec: str | None = None,
    feature_roots: tuple[Path, ...] = (),
    rerun: bool = False,
    prune_ratio: float = 0.0,
    prune_layer: int = 12,
) -> dict[str, Any]:
    percentage = validate_percentage(percentage)
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
    run_dir = run_directory(results_root, benchmark_name, model_name, method_name, seed)
    print(f"run directory: {run_dir}", flush=True)
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
        "training_free": True,
    }
    ensure_manifest(run_dir / "manifest.json", manifest)

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
        method = METHODS.create(method_name)
        # Batch CPU-only preprocessing (e.g. scene detection) runs before the model
        # backend is loaded, so it never competes with GPU memory.
        method.prepare(pending, run_dir / "prepare")
        model = MODELS.create(
            model_name,
            cache_dir=results_root / "cache",
            checkpoint=checkpoint,
            model_spec=model_spec,
            feature_roots=feature_roots,
            prune_ratio=prune_ratio,
            prune_layer=prune_layer,
        )
        method.validate_model(model)
        progress = tqdm(pending, desc=f"{benchmark_name}/{model_name}/{method_name}", unit="sample")
        for sample in progress:
            started = perf_counter()
            try:
                prediction = method.run(sample, model, run_dir / "cache" / sample.id)
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
    metrics_name = f"{percentage_key(percentage)}.json"
    write_json(run_dir / "metrics" / metrics_name, summary)
    # Export a submission whenever the benchmark has no local ground truth (hidden-label
    # splits like QVHighlights) or the full test set is evaluated. For hidden-label
    # benchmarks the submission is the only result, so it is produced at any subset.
    if (summary["metrics"] is None or percentage == 100) and summary["failed"] == 0:
        submission = benchmark.export_submission(
            values,
            results_root / "submissions" / benchmark_name / model_name / method_name / f"seed-{seed}.jsonl",
        )
        if submission is not None:
            summary["submission"] = str(submission)
            write_json(run_dir / "metrics" / metrics_name, summary)
    refresh_results_index(results_root)
    return summary
