"""Offline gates for staged, training-free inference optimizations."""

from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

from .metrics import evaluate


TPSA_V3_GATE = {
    "samples": 64,
    "EtF1_floor": 38.5851,
    "tF1@0.7_floor": 42.9489,
    "tIoU_floor": 57.8862,
    "minimum_primary_gain": 1.0,
    "C-Acc_floor": 48.4375,
    "parse_rate_floor": 1.0,
    "latency_ceiling_seconds": 6.754,
}


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


def validate_tpsa_v3_gate(
    control: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    control_metrics: dict[str, Any] | None = None,
    candidate_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the single predeclared OMTG-64 gate for repaired TPSA boundary."""
    control_metrics = dict(control_metrics or evaluate(control))
    candidate_metrics = dict(candidate_metrics or evaluate(candidate))
    control_by_id = _records_by_id(control)
    candidate_by_id = _records_by_id(candidate)
    same_ids = set(control_by_id) == set(candidate_by_id)
    ids = sorted(set(control_by_id) & set(candidate_by_id))
    equal_compute = same_ids
    for sample_id in ids:
        control_efficiency = control_by_id[sample_id].get("efficiency", {})
        candidate_efficiency = candidate_by_id[sample_id].get("efficiency", {})
        control_frames = control_efficiency.get("decoded_frames")
        candidate_frames = candidate_efficiency.get("decoded_frames")
        control_tokens = control_efficiency.get("actual_retained_tokens")
        candidate_tokens = candidate_efficiency.get("actual_retained_tokens")
        equal_compute = equal_compute and (
            control_frames is not None
            and control_tokens is not None
            and control_frames == candidate_frames
            and control_tokens == candidate_tokens
        )
    primary = ("EtF1", "tF1@0.7", "tIoU")
    floors = {
        name: float(candidate_metrics.get(name, float("-inf")))
        >= float(TPSA_V3_GATE[f"{name}_floor"])
        for name in primary
    }
    deltas = {
        name: float(candidate_metrics.get(name, float("-inf")))
        - float(control_metrics.get(name, float("inf")))
        for name in primary
    }
    primary_gain = max(deltas.values(), default=float("-inf"))
    latency = float(candidate_metrics.get(
        "mean_end_to_end_seconds", _mean_total_seconds(candidate),
    ))
    checks = {
        "exactly_64_samples": (
            len(control) == len(candidate) == len(ids) == TPSA_V3_GATE["samples"]
        ),
        "same_sample_ids": same_ids,
        "declared_policies": (
            all(row.get("spatial_policy") == "tpsa_query" for row in control)
            and all(row.get("spatial_policy") == "tpsa_boundary" for row in candidate)
        ),
        "primary_metric_floors": all(floors.values()),
        "one_point_primary_gain": primary_gain >= TPSA_V3_GATE["minimum_primary_gain"],
        "cardinality_floor": float(candidate_metrics.get("C-Acc", float("-inf")))
        >= TPSA_V3_GATE["C-Acc_floor"],
        "parse_rate": float(candidate_metrics.get("parse_rate", 0.0))
        >= TPSA_V3_GATE["parse_rate_floor"],
        "equal_frames_and_retained_tokens": equal_compute,
        "latency_ceiling": 0 < latency <= TPSA_V3_GATE["latency_ceiling_seconds"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(TPSA_V3_GATE),
        "matched_samples": len(ids),
        "metric_floors": floors,
        "primary_deltas": deltas,
        "largest_primary_gain": primary_gain,
        "candidate_mean_end_to_end_seconds": latency,
        "control_metrics": control_metrics,
        "candidate_metrics": candidate_metrics,
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
