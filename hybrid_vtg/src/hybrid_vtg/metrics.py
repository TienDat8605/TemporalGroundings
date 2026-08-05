"""Standard single-span VTG metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def temporal_iou(prediction: Sequence[float], target: Sequence[float]) -> float:
    overlap = max(0.0, min(float(prediction[1]), float(target[1])) - max(float(prediction[0]), float(target[0])))
    union = float(prediction[1]) - float(prediction[0]) + float(target[1]) - float(target[0]) - overlap
    return overlap / union if union > 0 else 0.0


def evaluate(records: Iterable[dict], thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict:
    ious = []
    route_ious = []
    boundary_errors = []
    retained_fractions = []
    token_ratios = []
    for record in records:
        targets = record.get("targets") or []
        if not targets or not record.get("prediction"):
            continue
        interval = record["prediction"]["interval"]
        ious.append(max(temporal_iou(interval, target) for target in targets))
        boundary_errors.append(min(
            (abs(interval[0] - target[0]) + abs(interval[1] - target[1])) / 2 for target in targets
        ))
        route = record.get("route") or {}
        components = route.get("components") or []
        route_ious.append(max(
            (temporal_iou((component["start"], component["end"]), target)
             for component in components for target in targets), default=0.0,
        ))
        if "retained_fraction" in route:
            retained_fractions.append(float(route["retained_fraction"]))
        efficiency = record.get("efficiency") or {}
        original_tokens = efficiency.get("semvid_original_tokens")
        retained_tokens = efficiency.get("semvid_retained_tokens")
        if original_tokens:
            token_ratios.append(float(retained_tokens) / float(original_tokens))
        else:
            stats = record["prediction"].get("semvid_stats") or {}
            if stats.get("orig_video_tokens"):
                token_ratios.append(float(stats["kept_video_tokens"]) / float(stats["orig_video_tokens"]))
    if not ious:
        return {"count": 0, "mIoU": 0.0, **{f"R@1,IoU={value:g}": 0.0 for value in thresholds}}
    return {
        "count": len(ious),
        "mIoU": sum(ious) / len(ious),
        "boundary_MAE_seconds": sum(boundary_errors) / len(boundary_errors),
        "mean_retained_duration_fraction": sum(retained_fractions) / len(retained_fractions) if retained_fractions else None,
        "mean_semvid_token_ratio": sum(token_ratios) / len(token_ratios) if token_ratios else None,
        **{f"R@1,IoU={threshold:g}": sum(iou >= threshold for iou in ious) / len(ious)
           for threshold in thresholds},
        **{f"RouterRecall@IoU={threshold:g}": sum(iou >= threshold for iou in route_ious) / len(route_ious)
           for threshold in thresholds},
    }
