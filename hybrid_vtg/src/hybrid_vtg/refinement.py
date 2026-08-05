"""High-FPS local boundary adjustment from frozen visual evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .coarse_encoder import FrozenSiglipEncoder, normalize_rows
from .config import RefinementConfig
from .types import Component, GroundingPrediction, Sample
from .video import decode_frames, sample_timestamps, visual_change


@dataclass(frozen=True)
class RefinedBoundary:
    interval: tuple[float, float]
    start_gain: float
    end_gain: float
    start_accepted: bool
    end_accepted: bool


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
        start_scores[index] = right_mean - left_mean + config.continuity_weight * change[index]
        end_scores[index] = left_mean - right_mean + config.continuity_weight * change[index]
    return start_scores, end_scores


def _best(times: np.ndarray, scores: np.ndarray, original: float, radius: float) -> tuple[float, float]:
    allowed = np.flatnonzero((times >= original - radius) & (times <= original + radius))
    if not len(allowed):
        return original, 0.0
    order = np.lexsort((times[allowed], np.abs(times[allowed] - original), -scores[allowed]))
    best = int(allowed[order[0]])
    nearest = int(np.argmin(np.abs(times - original)))
    return float(times[best]), float(scores[best] - scores[nearest])


def refine_from_signals(
    interval: Sequence[float], timestamps: Sequence[float], evidence: Sequence[float], continuity: Sequence[float],
    *, duration: float, component: Component, config: RefinementConfig,
) -> RefinedBoundary:
    start, end = map(float, interval)
    times = np.asarray(timestamps, dtype=float)
    start_scores, end_scores = boundary_scores(times, evidence, continuity, config)
    new_start, start_gain = _best(times, start_scores, start, config.radius_seconds)
    new_end, end_gain = _best(times, end_scores, end, config.radius_seconds)
    start_ok, end_ok = start_gain >= config.minimum_gain, end_gain >= config.minimum_gain
    refined_start = new_start if start_ok else start
    refined_end = new_end if end_ok else end
    refined_start = max(0.0, component.start, refined_start)
    refined_end = min(duration, component.end, refined_end)
    if refined_end <= refined_start:
        refined_start, refined_end, start_ok, end_ok = start, end, False, False
    return RefinedBoundary(
        interval=(round(refined_start, 3), round(refined_end, 3)),
        start_gain=start_gain, end_gain=end_gain, start_accepted=start_ok, end_accepted=end_ok,
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
    features = encoder.encode_images(decode_frames(sample.video_path, timestamps))
    evidence = normalize_rows(features) @ normalize_rows(query_embedding.reshape(1, -1))[0]
    continuity = visual_change(features)
    if len(timestamps) > 1:
        continuity[np.diff(timestamps, prepend=timestamps[0]) > 2.0 / config.fps] = 0.0
    return refine_from_signals(
        prediction.interval, timestamps, evidence, continuity, duration=sample.duration,
        component=prediction.component, config=config,
    )
