"""Single-span VTG and official OMTG set-valued metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .timestamps import consolidate_intervals, parse_intervals_detailed


def temporal_iou(prediction: Sequence[float], target: Sequence[float]) -> float:
    overlap = max(
        0.0,
        min(float(prediction[1]), float(target[1]))
        - max(float(prediction[0]), float(target[0])),
    )
    union = (
        float(prediction[1]) - float(prediction[0])
        + float(target[1]) - float(target[0]) - overlap
    )
    return overlap / union if union > 0 else 0.0


def one_to_many_metrics(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict[str, float]:
    """Official deterministic OMTG C-Acc, tIoU, Hungarian tF1, and EtF1."""
    prediction_values = [tuple(map(float, interval)) for interval in predictions]
    target_values = [tuple(map(float, interval)) for interval in targets]

    def merge(intervals: list[tuple[float, float]]) -> list[list[float]]:
        output: list[list[float]] = []
        for start, end in sorted(intervals):
            if output and start <= output[-1][1]:
                output[-1][1] = max(output[-1][1], end)
            else:
                output.append([start, end])
        return output

    prediction_union, target_union = merge(prediction_values), merge(target_values)
    prediction_length = sum(end - start for start, end in prediction_union)
    target_length = sum(end - start for start, end in target_union)
    intersection = sum(
        max(0.0, min(prediction_end, target_end) - max(prediction_start, target_start))
        for prediction_start, prediction_end in prediction_union
        for target_start, target_end in target_union
    )
    union = prediction_length + target_length - intersection
    result = {
        "C-Acc": float(len(prediction_values) == len(target_values)),
        "tIoU": intersection / union if union > 0 else 0.0,
        "CardinalityError": float(abs(len(prediction_values) - len(target_values))),
    }
    if not prediction_values or not target_values:
        value = float(not prediction_values and not target_values)
        for threshold in thresholds:
            result[f"tP@{threshold}"] = value
            result[f"tR@{threshold}"] = value
            result[f"tF1@{threshold}"] = value
        result["EtF1"] = value
        return result

    matrix = np.asarray([
        [temporal_iou(prediction, target) for target in target_values]
        for prediction in prediction_values
    ])
    prediction_indices, target_indices = linear_sum_assignment(-matrix)
    matched = matrix[prediction_indices, target_indices]
    for threshold in thresholds:
        true_positives = int((matched >= threshold).sum())
        precision = true_positives / len(prediction_values)
        recall = true_positives / len(target_values)
        result[f"tP@{threshold}"] = precision
        result[f"tR@{threshold}"] = recall
        result[f"tF1@{threshold}"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    result["EtF1"] = result["C-Acc"] * float(np.mean([
        result[f"tF1@{threshold}"] for threshold in thresholds
    ]))
    return result


def _efficiency_summary(records: Sequence[dict]) -> dict[str, float | None]:
    mappings = {
        "mean_visual_token_ratio": "effective_retention_ratio",
        "mean_decoded_frames": "decoded_frames",
        "mean_decoded_pixels": "decoded_pixels",
        "mean_vision_encoder_seconds": "vision_encoder_seconds",
        "mean_generation_seconds": "generation_seconds",
        "mean_prefill_tokens_before_pruning": "prefill_tokens_before_pruning",
        "mean_prefill_tokens_after_pruning": "prefill_tokens_after_pruning",
        "mean_end_to_end_seconds": "total_seconds",
    }
    output: dict[str, float | None] = {}
    for output_name, field in mappings.items():
        values = [
            float(record.get("efficiency", {})[field])
            for record in records if field in record.get("efficiency", {})
        ]
        output[output_name] = sum(values) / len(values) if values else None
    return output


def _prediction_intervals(record: dict) -> tuple[list[Sequence[float]], str]:
    prediction = record.get("prediction") or {}
    if not prediction:
        return [], "error"
    status = prediction.get("parse_status")
    if status:
        intervals = prediction.get("intervals")
        if intervals is None:
            intervals = [prediction["interval"]] if prediction.get("interval") else []
        return list(intervals), str(status)

    raw_text = prediction.get("raw_text")
    if raw_text is not None:
        try:
            parsed = parse_intervals_detailed(raw_text)
            intervals = consolidate_intervals(
                parsed.intervals,
                duration=float(record["duration"]),
            )
            return list(intervals), parsed.status
        except (KeyError, TypeError, ValueError):
            return [], "invalid"

    intervals = prediction.get("intervals")
    if intervals is not None:
        return list(intervals), "legacy"
    return ([prediction["interval"]] if prediction.get("interval") else []), "legacy"


def evaluate_omtg(records: Sequence[dict]) -> dict:
    labelled = [record for record in records if record.get("targets") is not None]
    parsed = [_prediction_intervals(record) for record in labelled]
    rows = [
        one_to_many_metrics(prediction, record.get("targets") or [])
        for record, (prediction, _) in zip(labelled, parsed)
    ]
    if not rows:
        return {"count": 0}
    parse_success = {"valid_json", "recovered", "explicit_empty", "parsed", "legacy"}
    status_counts = {
        status: sum(parsed_status == status for _, parsed_status in parsed)
        for status in sorted({parsed_status for _, parsed_status in parsed})
    }
    parsed_predictions = sum(status in parse_success for _, status in parsed)
    return {
        "count": len(rows),
        "metric_units": "percent except CardinalityError and efficiency counters",
        **{
            key: sum(row[key] for row in rows) / len(rows) * (
                1.0 if key == "CardinalityError" else 100.0
            )
            for key in rows[0]
        },
        "parsed_predictions": parsed_predictions,
        "parse_rate": parsed_predictions / len(labelled),
        "parse_status_counts": status_counts,
        **_efficiency_summary([record for record in labelled if record.get("prediction")]),
    }


def evaluate_single_span(
    records: Sequence[dict], thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict:
    labelled = [record for record in records if record.get("targets")]
    if not labelled:
        return {"count": 0, "mIoU": 0.0, **{
            f"R@1,IoU={value:g}": 0.0 for value in thresholds
        }}
    ious = [
        max(temporal_iou(record["prediction"]["interval"], target) for target in record["targets"])
        if record.get("prediction") and record["prediction"].get("interval") else 0.0
        for record in labelled
    ]
    boundary_errors = [
        min(
            (
                abs(float(record["prediction"]["interval"][0]) - float(target[0]))
                + abs(float(record["prediction"]["interval"][1]) - float(target[1]))
            ) / 2
            for target in record["targets"]
        )
        if record.get("prediction") and record["prediction"].get("interval")
        else float(record.get("duration", max(target[1] for target in record["targets"])))
        for record in labelled
    ]
    return {
        "count": len(labelled),
        "parsed_predictions": sum(
            bool(record.get("prediction") and record["prediction"].get("interval"))
            for record in labelled
        ),
        "parse_rate": sum(
            bool(record.get("prediction") and record["prediction"].get("interval"))
            for record in labelled
        ) / len(labelled),
        "mIoU": sum(ious) / len(ious),
        "boundary_MAE_seconds": sum(boundary_errors) / len(boundary_errors),
        **{
            f"R@1,IoU={threshold:g}": sum(iou >= threshold for iou in ious) / len(ious)
            for threshold in thresholds
        },
        **_efficiency_summary([record for record in labelled if record.get("prediction")]),
    }


def evaluate(records: Iterable[dict], thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict:
    values = list(records)
    if any(
        record.get("cardinality") == "multi" or record.get("group") == "omtg"
        for record in values
    ):
        return evaluate_omtg(values)
    return evaluate_single_span(values, thresholds)
