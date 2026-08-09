"""Offline gates for staged, training-free inference optimizations."""

from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

from .metrics import evaluate


def _records_by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in records}


def _intervals(record: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    prediction = record["prediction"]
    values = prediction.get("intervals")
    if values is None:
        values = [prediction["interval"]] if prediction.get("interval") else []
    return tuple((float(start), float(end)) for start, end in values)


def _mean_total_seconds(records: Sequence[dict[str, Any]]) -> float:
    values = [
        float(record.get("efficiency", {}).get("total_seconds", 0.0))
        for record in records
    ]
    values = [value for value in values if value > 0]
    return mean(values) if values else 0.0


def compare_optimization_runs(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    *,
    minimum_samples: int = 32,
    minimum_speedup: float = 0.0,
    logit_tolerance: float = 0.05,
    baseline_throughput: float = 0.0,
    candidate_throughput: float = 0.0,
) -> dict[str, Any]:
    """Compare fixed-fixture outputs without using labels to alter inference settings."""
    baseline_by_id = _records_by_id(baseline)
    candidate_by_id = _records_by_id(candidate)
    same_id_set = set(baseline_by_id) == set(candidate_by_id)
    ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    status_matches = sum(
        bool(baseline_by_id[sample_id].get("prediction"))
        == bool(candidate_by_id[sample_id].get("prediction"))
        for sample_id in ids
    )
    successful_ids = [
        sample_id for sample_id in ids
        if baseline_by_id[sample_id].get("prediction") and candidate_by_id[sample_id].get("prediction")
    ]
    baseline_rows = [baseline_by_id[sample_id] for sample_id in successful_ids]
    candidate_rows = [candidate_by_id[sample_id] for sample_id in successful_ids]

    exact_intervals = sum(
        _intervals(left) == _intervals(right) for left, right in zip(baseline_rows, candidate_rows)
    )
    endpoint_differences = [
        abs(a - b)
        for left, right in zip(baseline_rows, candidate_rows)
        for left_interval, right_interval in zip(_intervals(left), _intervals(right))
        for a, b in zip(left_interval, right_interval)
    ]
    baseline_metrics = evaluate(baseline_rows)
    candidate_metrics = evaluate(candidate_rows)
    baseline_seconds = _mean_total_seconds(baseline_rows)
    candidate_seconds = _mean_total_seconds(candidate_rows)
    if baseline_throughput > 0 and candidate_throughput > 0:
        speedup = candidate_throughput / baseline_throughput - 1.0
        timing_source = "run_wall_clock"
    else:
        speedup = baseline_seconds / candidate_seconds - 1.0 if baseline_seconds and candidate_seconds else 0.0
        timing_source = "mean_recorded_latency"
    oom_fallbacks = sum(
        int(bool(row.get("efficiency", {}).get("qwen_oom_fallback", False)))
        for row in candidate_rows
    )

    logit_rows = 0
    matching_top_tokens = 0
    maximum_logit_difference = 0.0
    for left, right in zip(baseline_rows, candidate_rows):
        left_top = left["prediction"].get("telemetry", {}).get("first_token_topk")
        right_top = right["prediction"].get("telemetry", {}).get("first_token_topk")
        if not left_top or not right_top:
            continue
        logit_rows += 1
        left_ids = list(left_top.get("token_ids", []))
        right_ids = list(right_top.get("token_ids", []))
        matching_top_tokens += bool(left_ids and right_ids and left_ids[0] == right_ids[0])
        if left_ids == right_ids:
            differences = [
                abs(float(a) - float(b))
                for a, b in zip(left_top.get("logits", []), right_top.get("logits", []))
            ]
            maximum_logit_difference = max([maximum_logit_difference, *differences])
        else:
            maximum_logit_difference = float("inf")

    equivalence_pass = (
        len(successful_ids) >= minimum_samples
        and same_id_set
        and status_matches == len(ids)
        and exact_intervals == len(successful_ids)
        and logit_rows == len(successful_ids)
        and matching_top_tokens == len(successful_ids)
        and maximum_logit_difference <= logit_tolerance
        and oom_fallbacks == 0
    )
    performance_pass = speedup >= minimum_speedup
    passed = equivalence_pass and performance_pass
    return {
        "passed": passed,
        "matched_samples": len(ids),
        "same_sample_ids": same_id_set,
        "successful_samples": len(successful_ids),
        "matching_parse_statuses": status_matches,
        "minimum_samples": minimum_samples,
        "exact_intervals": exact_intervals,
        "maximum_endpoint_difference_seconds": max(endpoint_differences, default=0.0),
        "logit_rows": logit_rows,
        "matching_first_tokens": matching_top_tokens,
        "maximum_top8_logit_difference": maximum_logit_difference,
        "logit_tolerance": logit_tolerance,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "baseline_mean_recorded_seconds": baseline_seconds,
        "candidate_mean_recorded_seconds": candidate_seconds,
        "speedup_fraction": speedup,
        "minimum_speedup_fraction": minimum_speedup,
        "timing_source": timing_source,
        "baseline_samples_per_second": baseline_throughput,
        "candidate_samples_per_second": candidate_throughput,
        "oom_fallbacks": oom_fallbacks,
        "equivalence_pass": equivalence_pass,
        "performance_pass": performance_pass,
    }


def validate_tpsa_promotion(
    semvid_runs: dict[str, Sequence[dict[str, Any]]],
    tpsa_runs: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Apply the predeclared 12.5%-token TPSA promotion gates across datasets."""
    datasets: dict[str, Any] = {}
    for name in sorted(set(semvid_runs) | set(tpsa_runs)):
        baseline_by_id = _records_by_id(semvid_runs.get(name, ()))
        candidate_by_id = _records_by_id(tpsa_runs.get(name, ()))
        same_ids = set(baseline_by_id) == set(candidate_by_id) and bool(baseline_by_id)
        ids = sorted(set(baseline_by_id) & set(candidate_by_id))
        matched = [
            sample_id for sample_id in ids
            if baseline_by_id[sample_id].get("prediction")
            and candidate_by_id[sample_id].get("prediction")
        ]
        baseline = [baseline_by_id[sample_id] for sample_id in matched]
        candidate = [candidate_by_id[sample_id] for sample_id in matched]
        baseline_metrics = evaluate(baseline)
        candidate_metrics = evaluate(candidate)
        recall_07_delta = float(
            candidate_metrics.get("R@1,IoU=0.7", 0.0)
            - baseline_metrics.get("R@1,IoU=0.7", 0.0)
        )
        recall_03_delta = float(
            candidate_metrics.get("R@1,IoU=0.3", 0.0)
            - baseline_metrics.get("R@1,IoU=0.3", 0.0)
        )
        baseline_mae = float(baseline_metrics.get("boundary_MAE_seconds", float("inf")))
        candidate_mae = float(candidate_metrics.get("boundary_MAE_seconds", float("inf")))
        mae_reduction = (
            (baseline_mae - candidate_mae) / baseline_mae
            if baseline_mae > 0 and baseline_mae != float("inf") else 0.0
        )
        baseline_seconds = _mean_total_seconds(baseline)
        candidate_seconds = _mean_total_seconds(candidate)
        overhead = (
            candidate_seconds / baseline_seconds - 1.0
            if baseline_seconds > 0 and candidate_seconds > 0 else float("inf")
        )
        equal_compute = all(
            left.get("efficiency", {}).get("decoded_frames")
            == right.get("efficiency", {}).get("decoded_frames")
            and left.get("efficiency", {}).get("actual_retained_tokens")
            == right.get("efficiency", {}).get("actual_retained_tokens")
            for left, right in zip(baseline, candidate)
        )
        primary_improvement = recall_07_delta >= 0.03 or mae_reduction >= 0.15
        safety_pass = recall_03_delta >= -0.01 and overhead < 0.10
        datasets[name] = {
            "same_sample_ids": same_ids,
            "matched_samples": len(matched),
            "equal_frames_and_tokens": equal_compute,
            "recall_07_delta": recall_07_delta,
            "boundary_mae_relative_reduction": mae_reduction,
            "recall_03_delta": recall_03_delta,
            "inference_overhead_fraction": overhead,
            "primary_improvement": primary_improvement,
            "safety_pass": safety_pass,
            "passed": same_ids and equal_compute and primary_improvement and safety_pass,
        }
    improved = sum(value["passed"] for value in datasets.values())
    return {
        "passed": improved >= 2,
        "improved_datasets": improved,
        "required_improved_datasets": 2,
        "datasets": datasets,
    }
