"""Single-run, resumable benchmark orchestration."""

from __future__ import annotations

import platform
from pathlib import Path
from time import perf_counter
from typing import Any

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
    existing = read_jsonl(predictions_path)
    done = {str(record["id"]) for record in existing}
    pending = [sample for sample in selected if sample.id not in done]
    if pending:
        method = METHODS.create(method_name)
        model = MODELS.create(
            model_name,
            cache_dir=results_root / "cache",
            checkpoint=checkpoint,
            model_spec=model_spec,
            feature_roots=feature_roots,
        )
        method.validate_model(model)
        for sample in pending:
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
                    },
                )
                continue
            append_jsonl(predictions_path, _record(sample, prediction, perf_counter() - started))

    records = read_jsonl(predictions_path)
    values = _evaluation_records(selected, records)
    successful_ids = {str(record["id"]) for record in records}
    metrics = benchmark.evaluate(values)
    summary = {
        "requested": len(selected),
        "successful": sum(sample.id in successful_ids for sample in selected),
        "failed": sum(sample.id not in successful_ids for sample in selected),
        "metrics": metrics,
    }
    metrics_name = f"{percentage_key(percentage)}.json"
    write_json(run_dir / "metrics" / metrics_name, summary)
    if percentage == 100:
        submission = benchmark.export_submission(
            values,
            results_root / "submissions" / benchmark_name / model_name / method_name / f"seed-{seed}.jsonl",
        )
        if submission is not None:
            summary["submission"] = str(submission)
            write_json(run_dir / "metrics" / metrics_name, summary)
    refresh_results_index(results_root)
    return summary
