"""Standard single-span VTG metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def temporal_iou(prediction: Sequence[float], target: Sequence[float]) -> float:
    overlap = max(0.0, min(float(prediction[1]), float(target[1])) - max(float(prediction[0]), float(target[0])))
    union = float(prediction[1]) - float(prediction[0]) + float(target[1]) - float(target[0]) - overlap
    return overlap / union if union > 0 else 0.0


def target_coverage(components: Sequence[dict], target: Sequence[float]) -> float:
    """Fraction of target duration available anywhere in the routed component union."""
    start, end = map(float, target)
    clipped = sorted(
        (max(start, float(item["start"])), min(end, float(item["end"])))
        for item in components
        if min(end, float(item["end"])) > max(start, float(item["start"]))
    )
    merged: list[list[float]] = []
    for lower, upper in clipped:
        if merged and lower <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], upper)
        else:
            merged.append([lower, upper])
    covered = sum(upper - lower for lower, upper in merged)
    return covered / (end - start) if end > start else 0.0


def endpoint_availability(components: Sequence[dict], target: Sequence[float]) -> tuple[bool, bool, bool]:
    start, end = map(float, target)
    start_available = any(float(item["start"]) <= start <= float(item["end"]) for item in components)
    end_available = any(float(item["start"]) <= end <= float(item["end"]) for item in components)
    contained = any(float(item["start"]) <= start and float(item["end"]) >= end for item in components)
    return start_available, end_available, contained


def evaluate(records: Iterable[dict], thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict:
    ious = []
    route_coverages = []
    route_start_available = []
    route_end_available = []
    route_containment = []
    boundary_errors = []
    retained_fractions = []
    token_ratios = []
    decoded_frames = []
    decoded_pixels = []
    vision_seconds = []
    prefill_before = []
    prefill_after = []
    total_seconds = []
    fallback_flags = []
    routed_component_counts = []
    component_rejection_fractions = []
    selected_presence_scores = []
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
        fallback_flags.append(bool(route.get("low_confidence_fallback", False)))
        routed_component_counts.append(len(components))
        component_predictions = record.get("component_predictions") or []
        component_errors = record.get("component_errors") or []
        attempted_components = len(component_predictions) + len(component_errors)
        if attempted_components:
            rejections = sum(error.get("event_present") is False for error in component_errors)
            component_rejection_fractions.append(rejections / attempted_components)
        if "presence_score" in record["prediction"]:
            selected_presence_scores.append(float(record["prediction"]["presence_score"]))
        target_route_values = []
        for target in targets:
            coverage = target_coverage(components, target)
            start_available, end_available, contained = endpoint_availability(components, target)
            target_route_values.append((coverage, start_available, end_available, contained))
        coverage, start_available, end_available, contained = max(
            target_route_values, key=lambda value: (value[0], value[3], value[1] and value[2]),
            default=(0.0, False, False, False),
        )
        route_coverages.append(coverage)
        route_start_available.append(start_available)
        route_end_available.append(end_available)
        route_containment.append(contained)
        if "retained_fraction" in route:
            retained_fractions.append(float(route["retained_fraction"]))
        efficiency = record.get("efficiency") or {}
        if "total_decoded_frames" in efficiency:
            decoded_frames.append(float(efficiency["total_decoded_frames"]))
        if "total_decoded_pixels" in efficiency:
            decoded_pixels.append(float(efficiency["total_decoded_pixels"]))
        if "total_vision_encoder_seconds" in efficiency:
            vision_seconds.append(float(efficiency["total_vision_encoder_seconds"]))
        if "llm_prefill_tokens_before_pruning" in efficiency:
            prefill_before.append(float(efficiency["llm_prefill_tokens_before_pruning"]))
        if "llm_prefill_tokens_after_pruning" in efficiency:
            prefill_after.append(float(efficiency["llm_prefill_tokens_after_pruning"]))
        timing = efficiency.get("timing_seconds") or {}
        if "total" in timing:
            total_seconds.append(float(timing["total"]))
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
        "mean_total_decoded_frames": sum(decoded_frames) / len(decoded_frames) if decoded_frames else None,
        "mean_total_decoded_pixels": sum(decoded_pixels) / len(decoded_pixels) if decoded_pixels else None,
        "mean_vision_encoder_seconds": sum(vision_seconds) / len(vision_seconds) if vision_seconds else None,
        "mean_llm_prefill_tokens_before_pruning": sum(prefill_before) / len(prefill_before) if prefill_before else None,
        "mean_llm_prefill_tokens_after_pruning": sum(prefill_after) / len(prefill_after) if prefill_after else None,
        "mean_end_to_end_seconds": sum(total_seconds) / len(total_seconds) if total_seconds else None,
        "TemporalFallbackRate": sum(fallback_flags) / len(fallback_flags),
        "mean_routed_component_count": sum(routed_component_counts) / len(routed_component_counts),
        "mean_component_rejection_fraction": (
            sum(component_rejection_fractions) / len(component_rejection_fractions)
            if component_rejection_fractions else None
        ),
        "mean_selected_presence_score": (
            sum(selected_presence_scores) / len(selected_presence_scores)
            if selected_presence_scores else None
        ),
        **{f"R@1,IoU={threshold:g}": sum(iou >= threshold for iou in ious) / len(ious)
           for threshold in thresholds},
        "RouterTargetCoverageMean": sum(route_coverages) / len(route_coverages),
        "RouterStartEndpointAvailable": sum(route_start_available) / len(route_start_available),
        "RouterEndEndpointAvailable": sum(route_end_available) / len(route_end_available),
        "RouterBothEndpointsAvailable": sum(
            start and end for start, end in zip(route_start_available, route_end_available)
        ) / len(route_start_available),
        "RouterFullContainment": sum(route_containment) / len(route_containment),
        **{f"RouterTargetCoverage@{threshold:g}": sum(value >= threshold for value in route_coverages) / len(route_coverages)
           for threshold in thresholds},
    }
