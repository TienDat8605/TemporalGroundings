"""High-FPS local boundary adjustment from frozen visual evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .coarse_encoder import FrozenSiglipEncoder, normalize_rows
from .config import RefinementConfig
from .types import Component, GroundingPrediction, Sample
from .video import decode_frames, sample_timestamps, visual_change


@dataclass(frozen=True)
class RefinedBoundary:
    interval: tuple[float, float]
    joint_gain: float
    interval_contrast: float
    duration_ratio: float
    start_gain: float
    end_gain: float
    start_accepted: bool
    end_accepted: bool
    decoded_frames: int = 0
    decoded_pixels: int = 0
    processor_seconds: float = 0.0
    vision_encoder_seconds: float = 0.0


@dataclass(frozen=True)
class RefinementDecision:
    refine: bool
    tier: str
    confidence: float
    fps: float
    reason: str


def decide_refinement(
    prediction: GroundingPrediction,
    quality: Mapping[str, float],
    config: RefinementConfig,
    *,
    expert_fps: float,
    low_confidence_route: bool,
) -> RefinementDecision:
    """Choose a fixed, label-free refinement budget from endpoint evidence."""
    if not config.enabled:
        return RefinementDecision(False, "disabled", 0.0, 0.0, "refinement_disabled")
    if not config.adaptive:
        return RefinementDecision(True, "fixed", 0.0, config.fps, "fixed_fps")

    confidence = float(quality.get("boundary_confidence", 0.0))
    start, end = prediction.interval
    frame_period = 1.0 / expert_fps
    clear_of_edges = (
        start - prediction.component.start > frame_period
        and prediction.component.end - end > frame_period
    )
    safe_geometry = clear_of_edges and not low_confidence_route
    if safe_geometry and confidence >= config.high_confidence:
        return RefinementDecision(False, "high", confidence, 0.0, "strong_endpoint_contrast")
    if safe_geometry and confidence >= config.medium_confidence:
        return RefinementDecision(True, "medium", confidence, config.medium_fps, "moderate_endpoint_contrast")
    reason = "component_edge" if not clear_of_edges else (
        "low_confidence_route" if low_confidence_route else "weak_endpoint_contrast"
    )
    return RefinementDecision(True, "low", confidence, config.low_fps, reason)


def boundary_scores(
    timestamps: Sequence[float], evidence: Sequence[float], continuity: Sequence[float], config: RefinementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps, dtype=float)
    values = np.asarray(evidence, dtype=float)
    change = np.asarray(continuity, dtype=float)
    if not len(times) or len(times) != len(values) or len(times) != len(change):
        raise ValueError("boundary signals must be equal non-empty vectors")
    step = float(np.median(np.diff(times))) if len(times) > 1 else config.evidence_window_seconds
    radius = max(1, int(round(config.evidence_window_seconds / max(step, 1e-6))))
    start_scores, end_scores = np.zeros(len(times)), np.zeros(len(times))
    for index in range(len(times)):
        left = values[max(0, index - radius):index]
        right = values[index:min(len(times), index + radius)]
        left_mean = float(left.mean()) if len(left) else float(values[index])
        right_mean = float(right.mean()) if len(right) else float(values[index])
        start_contrast = right_mean - left_mean
        end_contrast = left_mean - right_mean
        # A shot cut is useful only when it agrees with the direction of query evidence.
        start_scores[index] = start_contrast + config.continuity_weight * change[index] * max(start_contrast, 0.0)
        end_scores[index] = end_contrast + config.continuity_weight * change[index] * max(end_contrast, 0.0)
    return start_scores, end_scores


def _mean(values: np.ndarray, mask: np.ndarray, fallback: float) -> float:
    selected = values[mask]
    return float(selected.mean()) if len(selected) else fallback


def refine_from_signals(
    interval: Sequence[float], timestamps: Sequence[float], evidence: Sequence[float], continuity: Sequence[float],
    *, duration: float, component: Component, config: RefinementConfig,
) -> RefinedBoundary:
    start, end = map(float, interval)
    times = np.asarray(timestamps, dtype=float)
    values = np.asarray(evidence, dtype=float)
    start_scores, end_scores = boundary_scores(times, evidence, continuity, config)
    start_indices = np.flatnonzero((times >= start - config.radius_seconds) & (times <= start + config.radius_seconds))
    end_indices = np.flatnonzero((times >= end - config.radius_seconds) & (times <= end + config.radius_seconds))
    original_duration = max(end - start, 1e-6)

    def pair_score(start_index: int, end_index: int) -> tuple[float, float, float]:
        candidate_start, candidate_end = float(times[start_index]), float(times[end_index])
        candidate_duration = candidate_end - candidate_start
        if candidate_duration <= 0:
            return -np.inf, -np.inf, 0.0
        inside_mask = (times >= candidate_start) & (times <= candidate_end)
        outside_mask = (
            ((times >= candidate_start - config.evidence_window_seconds) & (times < candidate_start))
            | ((times > candidate_end) & (times <= candidate_end + config.evidence_window_seconds))
        )
        inside_mean = _mean(values, inside_mask, float(values.mean()))
        outside_mean = _mean(values, outside_mask, float(values.mean()))
        contrast = inside_mean - outside_mean
        duration_ratio = candidate_duration / original_duration
        duration_penalty = abs(float(np.log(max(duration_ratio, 1e-6))))
        score = (
            float(start_scores[start_index]) + float(end_scores[end_index])
            + config.inside_contrast_weight * contrast
            - config.duration_prior_weight * duration_penalty
        )
        return score, contrast, duration_ratio

    nearest_start = int(np.argmin(np.abs(times - start)))
    nearest_end = int(np.argmin(np.abs(times - end)))
    baseline_score, _, _ = pair_score(nearest_start, nearest_end)
    choices = []
    for start_index in start_indices:
        for end_index in end_indices:
            score, contrast, duration_ratio = pair_score(int(start_index), int(end_index))
            if not np.isfinite(score):
                continue
            displacement = abs(float(times[start_index]) - start) + abs(float(times[end_index]) - end)
            choices.append((score, -displacement, -float(times[start_index]), int(start_index), int(end_index), contrast, duration_ratio))
    best = max(choices, default=None)
    joint_gain = float(best[0] - baseline_score) if best is not None else 0.0
    accepted = best is not None and joint_gain >= config.minimum_gain
    if accepted:
        refined_start = max(0.0, component.start, float(times[best[3]]))
        refined_end = min(duration, component.end, float(times[best[4]]))
        interval_contrast, duration_ratio = float(best[5]), float(best[6])
    else:
        refined_start, refined_end = start, end
        _, interval_contrast, duration_ratio = pair_score(nearest_start, nearest_end)
    if refined_end <= refined_start:
        refined_start, refined_end, accepted = start, end, False
    start_gain = float(start_scores[best[3]] - start_scores[nearest_start]) if accepted else 0.0
    end_gain = float(end_scores[best[4]] - end_scores[nearest_end]) if accepted else 0.0
    return RefinedBoundary(
        interval=(round(refined_start, 3), round(refined_end, 3)),
        joint_gain=joint_gain, interval_contrast=float(interval_contrast), duration_ratio=float(duration_ratio),
        start_gain=start_gain, end_gain=end_gain, start_accepted=accepted, end_accepted=accepted,
    )


def refine_prediction(
    sample: Sample,
    prediction: GroundingPrediction,
    encoder: FrozenSiglipEncoder,
    query_embedding: np.ndarray,
    config: RefinementConfig,
) -> RefinedBoundary:
    start, end = prediction.interval
    neighborhoods = (
        (max(prediction.component.start, start - config.radius_seconds),
         min(prediction.component.end, start + config.radius_seconds)),
        (max(prediction.component.start, end - config.radius_seconds),
         min(prediction.component.end, end + config.radius_seconds)),
    )
    timestamp_parts = [
        sample_timestamps(sample.duration, config.fps, start=lower, end=upper)
        for lower, upper in neighborhoods if upper > lower
    ]
    timestamps = np.unique(np.concatenate(timestamp_parts))
    frames = decode_frames(sample.video_path, timestamps)
    decoded_pixels = sum(image.width * image.height for image in frames)
    features = encoder.encode_images(frames)
    encode_stats = encoder.last_encode_stats
    evidence = normalize_rows(features) @ normalize_rows(query_embedding.reshape(1, -1))[0]
    continuity = visual_change(features)
    if len(timestamps) > 1:
        continuity[np.diff(timestamps, prepend=timestamps[0]) > 2.0 / config.fps] = 0.0
    result = refine_from_signals(
        prediction.interval, timestamps, evidence, continuity, duration=sample.duration,
        component=prediction.component, config=config,
    )
    return replace(
        result,
        decoded_frames=len(frames),
        decoded_pixels=decoded_pixels,
        processor_seconds=float(encode_stats.get("processor_seconds", 0.0)),
        vision_encoder_seconds=float(encode_stats.get("vision_encoder_seconds", 0.0)),
    )
