"""Temporal-grounding metrics for the canonical result schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def temporal_iou(prediction: Sequence[float], target: Sequence[float]) -> float:
    overlap = max(0.0, min(float(prediction[1]), float(target[1])) - max(float(prediction[0]), float(target[0])))
    union = float(prediction[1]) - float(prediction[0]) + float(target[1]) - float(target[0]) - overlap
    return overlap / union if union > 0 else 0.0


def _spans(record: Mapping) -> list[list[float]]:
    return [[float(value["start"]), float(value["end"])] for value in (record.get("prediction") or {}).get("spans", [])]


def _targets(record: Mapping) -> list[list[float]]:
    return [[float(value[0]), float(value[1])] for value in record.get("targets", [])]


def one_to_many_metrics(predictions, targets, thresholds=(0.3, 0.5, 0.7)) -> dict[str, float]:
    def merge(values):
        output = []
        for start, end in sorted(values):
            if output and start <= output[-1][1]:
                output[-1][1] = max(output[-1][1], end)
            else:
                output.append([start, end])
        return output

    prediction_union, target_union = merge(predictions), merge(targets)
    predicted_length = sum(end - start for start, end in prediction_union)
    target_length = sum(end - start for start, end in target_union)
    intersection = sum(max(0.0, min(pe, te) - max(ps, ts)) for ps, pe in prediction_union for ts, te in target_union)
    union = predicted_length + target_length - intersection
    output = {
        "C-Acc": float(len(predictions) == len(targets)),
        "tIoU": intersection / union if union else 0.0,
        "CardinalityError": float(abs(len(predictions) - len(targets))),
    }
    if predictions and targets:
        matrix = np.asarray([[temporal_iou(prediction, target) for target in targets] for prediction in predictions])
        rows, columns = linear_sum_assignment(-matrix)
        matched = matrix[rows, columns]
    else:
        matched = np.asarray([])
    for threshold in thresholds:
        true_positive = int((matched >= threshold).sum())
        precision = true_positive / len(predictions) if predictions else 0.0
        recall = true_positive / len(targets) if targets else 0.0
        output[f"tP@{threshold}"] = precision
        output[f"tR@{threshold}"] = recall
        output[f"tF1@{threshold}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    output["EtF1"] = output["C-Acc"] * float(np.mean([output[f"tF1@{value}"] for value in thresholds]))
    return output


def evaluate_records(records: Sequence[Mapping], *, multi_span: bool) -> dict[str, float | int]:
    labelled = [record for record in records if record.get("targets")]
    if not labelled:
        return {"count": 0}
    if multi_span:
        rows = [one_to_many_metrics(_spans(record), _targets(record)) for record in labelled]
        return {
            "count": len(rows),
            **{
                key: sum(row[key] for row in rows) / len(rows) * (1.0 if key == "CardinalityError" else 100.0)
                for key in rows[0]
            },
        }
    ious = [
        max((temporal_iou(predictions[0], target) for target in _targets(record)), default=0.0)
        if (predictions := _spans(record))
        else 0.0
        for record in labelled
    ]
    return {
        "count": len(labelled),
        "mIoU": sum(ious) / len(ious),
        **{
            f"R@1,IoU={threshold:g}": sum(value >= threshold for value in ious) / len(ious)
            for threshold in (0.3, 0.5, 0.7)
        },
    }
