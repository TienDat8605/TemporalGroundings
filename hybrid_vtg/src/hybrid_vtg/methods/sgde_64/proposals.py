"""Candidate proposal extraction for SGDE (Idea 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...postprocess import temporal_iou
from .scout import ScoutTimeline


@dataclass(frozen=True)
class CandidateProposal:
    start: float
    end: float
    peak_z: float
    mean_z: float
    penalized_score: float
    score: float
    source: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, float | str]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "peak_z": round(self.peak_z, 3),
            "mean_z": round(self.mean_z, 3),
            "penalized_score": round(self.penalized_score, 3),
            "score": round(self.score, 3),
            "source": self.source,
        }


def _interval_metrics(
    timestamps: np.ndarray,
    z_scores: np.ndarray,
    start: float,
    end: float,
    tau: float = 0.3,
    lambda_len: float = 0.01,
) -> tuple[float, float, float]:
    """Compute peak_z, mean_z, and penalized_score J(a, b) for a time interval [start, end]."""
    mask = (timestamps >= start - 1e-4) & (timestamps <= end + 1e-4)
    dt = float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    if not np.any(mask):
        idx = int(np.argmin(np.abs(timestamps - (start + end) / 2.0)))
        peak_z = float(z_scores[idx])
        mean_z = peak_z
        j_score = float((peak_z - tau - lambda_len) * max(dt, float(end - start)))
        return peak_z, mean_z, j_score

    selected = z_scores[mask]
    peak_z = float(np.max(selected))
    mean_z = float(np.mean(selected))
    penalized = float(np.sum(selected - tau - lambda_len) * dt)
    return peak_z, mean_z, penalized


def _composite_score(peak_z: float, mean_z: float, penalized_score: float, duration: float) -> float:
    """Combine peak and positive energy without penalizing natural event duration."""
    excess_energy = max(0.0, penalized_score)
    return float(peak_z + 0.5 * excess_energy + 0.2 * max(0.0, mean_z))


def extract_hysteresis_components(
    timestamps: np.ndarray,
    z_scores: np.ndarray,
    *,
    high_threshold: float = 0.8,
    low_threshold: float = 0.25,
    min_duration: float = 3.0,
) -> list[CandidateProposal]:
    """Extract connected components where score reaches high_threshold and stays above low_threshold."""
    if len(timestamps) == 0:
        return []

    proposals: list[CandidateProposal] = []
    n = len(timestamps)
    in_component = False
    start_idx = 0
    has_high = False

    for i in range(n):
        z = z_scores[i]
        if not in_component:
            if z >= low_threshold:
                in_component = True
                start_idx = i
                has_high = (z >= high_threshold)
        else:
            if z >= low_threshold:
                if z >= high_threshold:
                    has_high = True
            else:
                # Component ended
                in_component = False
                if has_high:
                    end_idx = i - 1
                    t_start = float(timestamps[start_idx])
                    t_end = float(timestamps[end_idx])
                    dur = max(t_end - t_start, min_duration)
                    t_end = t_start + dur
                    peak_z, mean_z, pen = _interval_metrics(timestamps, z_scores, t_start, t_end)
                    score = _composite_score(peak_z, mean_z, pen, dur)
                    proposals.append(CandidateProposal(t_start, t_end, peak_z, mean_z, pen, score, "hysteresis"))

    # Tail component
    if in_component and has_high:
        t_start = float(timestamps[start_idx])
        t_end = float(timestamps[n - 1])
        dur = max(t_end - t_start, min_duration)
        t_end = t_start + dur
        peak_z, mean_z, pen = _interval_metrics(timestamps, z_scores, t_start, t_end)
        score = _composite_score(peak_z, mean_z, pen, dur)
        proposals.append(CandidateProposal(t_start, t_end, peak_z, mean_z, pen, score, "hysteresis"))

    return proposals


def extract_penalized_intervals(
    timestamps: np.ndarray,
    z_scores: np.ndarray,
    *,
    tau: float = 0.5,
    lambda_len: float = 0.05,
    min_duration: float = 1.0,
    max_duration: float = 60.0,
) -> list[CandidateProposal]:
    """Find locally maximal sub-intervals maximizing J(a,b) = integral (z - tau) - lambda*(b-a)."""
    if len(timestamps) <= 1:
        return []

    proposals: list[CandidateProposal] = []
    # Identify local peaks
    peak_indices = []
    for i in range(len(z_scores)):
        prev_z = z_scores[i - 1] if i > 0 else -1e9
        next_z = z_scores[i + 1] if i < len(z_scores) - 1 else -1e9
        if z_scores[i] >= prev_z and z_scores[i] >= next_z and z_scores[i] >= tau:
            peak_indices.append(i)

    # For each peak, expand left and right to maximize J
    dt = float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    for peak in peak_indices:
        best_left = peak
        best_right = peak
        best_j = (float(z_scores[peak]) - tau) * dt - lambda_len * dt

        # Expand window
        left = peak
        right = peak
        while left > 0 or right < len(timestamps) - 1:
            dur = timestamps[right] - timestamps[left]
            if dur >= max_duration:
                break
            expanded = False
            if left > 0:
                cand_left = left - 1
                peak_z, mean_z, j_cand = _interval_metrics(
                    timestamps, z_scores, timestamps[cand_left], timestamps[right], tau, lambda_len
                )
                if j_cand >= best_j:
                    best_j = j_cand
                    best_left = cand_left
                    left = cand_left
                    expanded = True
            if right < len(timestamps) - 1:
                cand_right = right + 1
                peak_z, mean_z, j_cand = _interval_metrics(
                    timestamps, z_scores, timestamps[best_left], timestamps[cand_right], tau, lambda_len
                )
                if j_cand >= best_j:
                    best_j = j_cand
                    best_right = cand_right
                    right = cand_right
                    expanded = True
            if not expanded:
                break

        t_start = float(timestamps[best_left])
        t_end = float(timestamps[best_right])
        t_end = max(t_end, t_start + min_duration)
        peak_z, mean_z, pen = _interval_metrics(timestamps, z_scores, t_start, t_end, tau, lambda_len)
        if pen > 0:
            score = _composite_score(peak_z, mean_z, pen, t_end - t_start)
            proposals.append(CandidateProposal(t_start, t_end, peak_z, mean_z, pen, score, "penalized_interval"))

    return proposals


def extract_multiscale_density_windows(
    timestamps: np.ndarray,
    z_scores: np.ndarray,
    video_duration: float,
    *,
    scales: Sequence[float] = (6.0, 12.0, 24.0, 48.0),
    min_peak_z: float = 0.5,
) -> list[CandidateProposal]:
    """Find local score peaks at several candidate durations."""
    if len(timestamps) == 0:
        return []

    proposals: list[CandidateProposal] = []
    for scale in scales:
        if scale > video_duration * 0.9 and scale > 5.0:
            continue
        half = scale / 2.0
        # Evaluate density centered at timestamps
        for i, center in enumerate(timestamps):
            if z_scores[i] < min_peak_z:
                continue
            start = max(0.0, center - half)
            end = min(video_duration, center + half)
            if end - start < 1.0:
                continue
            peak_z, mean_z, pen = _interval_metrics(timestamps, z_scores, start, end)
            if peak_z >= min_peak_z:
                score = _composite_score(peak_z, mean_z, pen, end - start)
                proposals.append(CandidateProposal(start, end, peak_z, mean_z, pen, score, f"multiscale_{int(scale)}s"))

    return proposals


def temporal_nms(
    candidates: Sequence[CandidateProposal],
    *,
    iou_threshold: float = 0.5,
    max_candidates: int = 4,
) -> list[CandidateProposal]:
    """Apply 1D temporal NMS to retain diverse high-scoring proposals."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: (-c.score, -c.peak_z, c.start))
    selected: list[CandidateProposal] = []

    for cand in sorted_cands:
        overlap = False
        for kept in selected:
            iou = temporal_iou(cand, kept)
            if iou >= iou_threshold:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
            if len(selected) >= max_candidates:
                break

    return sorted(selected, key=lambda c: c.start)


def extract_candidate_proposals(
    timeline: ScoutTimeline,
    duration: float,
    *,
    cardinality: str = "single",
    nms_iou: float = 0.5,
    max_multi_candidates: int = 4,
) -> tuple[list[CandidateProposal], bool]:
    """Generate and rank complementary candidate proposals, returning (selected, is_confident)."""
    if len(timeline) == 0 or duration <= 0:
        return [], False

    # Check if timeline has any valid contrast
    if timeline.peak_z < 0.5 or timeline.mad < 1e-4:
        return [], False

    all_proposals: list[CandidateProposal] = []

    # 1. Hysteresis connected components
    hysteresis = extract_hysteresis_components(
        timeline.timestamps,
        timeline.z_scores,
        high_threshold=1.0,
        low_threshold=0.3,
    )
    all_proposals.extend(hysteresis)

    # 2. Penalized intervals
    penalized = extract_penalized_intervals(
        timeline.timestamps,
        timeline.z_scores,
        tau=0.5,
        lambda_len=0.05,
    )
    all_proposals.extend(penalized)

    # 3. Multi-scale density windows
    multiscale = extract_multiscale_density_windows(
        timeline.timestamps,
        timeline.z_scores,
        duration,
        scales=(2.0, 5.0, 10.0, 20.0, 40.0),
    )
    all_proposals.extend(multiscale)

    if not all_proposals:
        return [], False

    max_k = 1 if cardinality == "single" else max_multi_candidates
    selected = temporal_nms(all_proposals, iou_threshold=nms_iou, max_candidates=max_k)

    # Confidence check
    is_confident = (len(selected) > 0 and selected[0].score >= 0.4 and timeline.peak_z >= 0.8)
    return selected, is_confident
