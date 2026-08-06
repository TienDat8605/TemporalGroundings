"""Offline gates for staged, training-free inference optimizations."""

from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

from .metrics import evaluate


def _records_by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in records}


def _interval(record: dict[str, Any]) -> tuple[float, float]:
    start, end = record["prediction"]["interval"]
    return float(start), float(end)


def _mean_total_seconds(records: Sequence[dict[str, Any]]) -> float:
    values = [
        float(record.get("efficiency", {}).get("timing_seconds", {}).get("total", 0.0))
        for record in records
    ]
    values = [value for value in values if value > 0]
    return mean(values) if values else 0.0


def compare_optimization_runs(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    *,
    mode: str,
    minimum_samples: int = 32,
    minimum_speedup: float = 0.0,
    logit_tolerance: float = 0.05,
    baseline_throughput: float = 0.0,
    candidate_throughput: float = 0.0,
) -> dict[str, Any]:
    """Compare fixed-fixture outputs without using labels to alter inference settings."""
    if mode not in {"equivalence", "refinement", "combined"}:
        raise ValueError("validation mode must be equivalence, refinement, or combined")
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
        _interval(left) == _interval(right) for left, right in zip(baseline_rows, candidate_rows)
    )
    endpoint_differences = [
        abs(a - b)
        for left, right in zip(baseline_rows, candidate_rows)
        for a, b in zip(_interval(left), _interval(right))
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
        int(row.get("efficiency", {}).get("qwen_oom_fallbacks", 0)) for row in candidate_rows
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

    recall_keys = [key for key in baseline_metrics if key.startswith("R@1,")]
    recall_deltas = {
        key: float(candidate_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0))
        for key in recall_keys
    }
    accuracy_pass = (
        float(candidate_metrics.get("mIoU", 0.0))
        >= float(baseline_metrics.get("mIoU", 0.0)) - 0.005
        and all(delta >= -0.01 for delta in recall_deltas.values())
    )
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
    passed = (
        equivalence_pass and performance_pass
        if mode == "equivalence"
        else (
            len(successful_ids) >= minimum_samples and same_id_set and status_matches == len(ids)
            and accuracy_pass and performance_pass and oom_fallbacks == 0
        )
    )
    return {
        "passed": passed,
        "mode": mode,
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
        "recall_deltas": recall_deltas,
        "mIoU_delta": float(candidate_metrics.get("mIoU", 0.0) - baseline_metrics.get("mIoU", 0.0)),
        "baseline_mean_recorded_seconds": baseline_seconds,
        "candidate_mean_recorded_seconds": candidate_seconds,
        "speedup_fraction": speedup,
        "minimum_speedup_fraction": minimum_speedup,
        "timing_source": timing_source,
        "baseline_samples_per_second": baseline_throughput,
        "candidate_samples_per_second": candidate_throughput,
        "oom_fallbacks": oom_fallbacks,
        "equivalence_pass": equivalence_pass,
        "accuracy_pass": accuracy_pass,
        "performance_pass": performance_pass,
    }
