"""Training-free boundary-guided temporal evidence sparsification."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample, TemporalEvidence
from ...media import uniform_timestamps
from ..budget import (
    CELL_SECONDS,
    BudgetLedger,
    duplicate_tubelets,
    duration_budget,
    pad_even,
    requires_even_frames,
    scout_timestamps,
    temporal_anchor_indices,
)
from ..hmve import pack_evidence

PERSISTENCE = 2
ABSENT_THRESHOLD = 0.0
PRESENT_THRESHOLD = 1.0
MAXIMUM_PAIRS = 6
TARGET_RESOLUTION = 1.0
AMBIGUITY_PROBE_ROUNDS = 1
RETENTION_RATIO = 0.125


@dataclass(frozen=True)
class PresenceCalibration:
    median: float
    mad: float
    scale: float
    constant: bool = False


@dataclass(frozen=True)
class CoarseObservation:
    timestamp: float
    raw_scores: tuple[float, ...]
    aggregate: float
    normalized: float | None = None
    state: str = "uncertain"
    row_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class StateRun:
    state: str
    start: float
    end: float


@dataclass(frozen=True)
class BoundaryBracket:
    start: float
    end: float
    left_state: str
    right_state: str
    role: str
    confidence: float
    edge: bool = False


@dataclass(frozen=True)
class BoundaryCorridor:
    start: float
    end: float
    role: str
    status: str
    reason: str
    confidence: float
    pair_id: int


@dataclass(frozen=True)
class EpisodePair:
    pair_id: int
    start: BoundaryBracket
    end: BoundaryBracket
    score: float

    @property
    def interval(self) -> tuple[float, float]:
        start = 0.0 if self.start.edge else self.start.end
        end = self.end.end if self.end.edge else self.end.start
        return start, end


@dataclass
class _ActiveBoundary:
    pair_id: int
    role: str
    left: float
    right: float
    left_state: str
    right_state: str
    confidence: float
    maximum_depth: int
    depth: int = 0
    ambiguity_probes: int = 0


def aggregate_query_presence(evidence: TemporalEvidence, scores) -> tuple[CoarseObservation, ...]:
    """Aggregate spatial rows at each timestamp with an upper-quartile mean."""
    grouped: dict[float, list[tuple[float, int]]] = {}
    for index, (timestamp, score) in enumerate(zip(evidence.timestamps, scores.tolist())):
        grouped.setdefault(round(float(timestamp), 6), []).append((float(score), index))
    observations = []
    for timestamp, values in sorted(grouped.items()):
        ranked = sorted((value for value, _ in values), reverse=True)
        count = max(1, math.ceil(len(ranked) / 4))
        observations.append(
            CoarseObservation(
                timestamp=timestamp,
                raw_scores=tuple(value for value, _ in values),
                aggregate=sum(ranked[:count]) / count,
                row_indices=tuple(index for _, index in values),
            )
        )
    return tuple(observations)


def normalize_presence(
    observations: Sequence[CoarseObservation],
) -> tuple[tuple[CoarseObservation, ...], PresenceCalibration]:
    values = [value.aggregate for value in observations]
    if not values:
        raise ValueError("presence timeline cannot be empty")
    center = float(statistics.median(values))
    deviations = [abs(value - center) for value in values]
    mad = float(statistics.median(deviations))
    constant = max(values) - min(values) <= 1e-12
    positive = [value for value in deviations if value > 1e-12]
    scale = mad if mad > 1e-12 else (min(positive) if positive else 1.0)
    calibration = PresenceCalibration(center, mad, scale, constant)
    normalized = []
    for observation in observations:
        if constant:
            normalized.append(replace(observation, normalized=None, state="uncertain"))
            continue
        z_score = (observation.aggregate - center) / scale
        state = _state_for_z(z_score)
        normalized.append(replace(observation, normalized=z_score, state=state))
    return tuple(normalized), calibration


def _state_for_z(value: float) -> str:
    if value <= ABSENT_THRESHOLD:
        return "absent"
    if value >= PRESENT_THRESHOLD:
        return "present"
    return "uncertain"


def state_for_score(score: float, calibration: PresenceCalibration) -> tuple[float | None, str]:
    if calibration.constant:
        return None, "uncertain"
    value = (score - calibration.median) / calibration.scale
    return value, _state_for_z(value)


def detect_boundaries(
    observations: Sequence[CoarseObservation], duration: float
) -> tuple[tuple[BoundaryBracket, ...], tuple[StateRun, ...]]:
    """Confirm state changes only after two adjacent equal confident states."""
    brackets: list[BoundaryBracket] = []
    runs: list[StateRun] = []
    current: str | None = None
    current_start = 0.0
    last_current: CoarseObservation | None = None
    streak_state: str | None = None
    streak: list[CoarseObservation] = []

    for observation in observations:
        if observation.state == "uncertain":
            streak_state, streak = None, []
            continue
        if observation.state == streak_state:
            streak.append(observation)
        else:
            streak_state, streak = observation.state, [observation]
        if current == observation.state:
            last_current = observation
        if len(streak) < PERSISTENCE or current == observation.state:
            continue

        confirmed = streak[0]
        if current is None:
            current = observation.state
            current_start = confirmed.timestamp
            last_current = observation
            if current == "present":
                brackets.append(
                    BoundaryBracket(0.0, 0.0, "edge", "present", "start", _magnitude(confirmed), edge=True)
                )
            continue

        assert last_current is not None
        role = "start" if current == "absent" else "end"
        confidence = abs(_normalized(confirmed) - _normalized(last_current))
        brackets.append(
            BoundaryBracket(
                last_current.timestamp,
                confirmed.timestamp,
                current,
                observation.state,
                role,
                confidence,
            )
        )
        runs.append(StateRun(current, current_start, last_current.timestamp))
        current = observation.state
        current_start = confirmed.timestamp
        last_current = observation

    if current is not None and last_current is not None:
        runs.append(StateRun(current, current_start, last_current.timestamp))
        if current == "present":
            brackets.append(
                BoundaryBracket(duration, duration, "present", "edge", "end", _magnitude(last_current), edge=True)
            )
    return tuple(brackets), tuple(runs)


def pair_boundaries(
    brackets: Sequence[BoundaryBracket], observations: Sequence[CoarseObservation], duration: float
) -> tuple[EpisodePair, ...]:
    pairs: list[EpisodePair] = []
    pending: BoundaryBracket | None = None
    for bracket in sorted(brackets, key=lambda value: (value.start, value.end, value.role)):
        if bracket.role == "start":
            pending = bracket
        elif pending is not None:
            provisional = EpisodePair(len(pairs), pending, bracket, 0.0)
            start, end = provisional.interval
            if end > start:
                inside = [_normalized(value) for value in observations if start <= value.timestamp <= end]
                before = [_normalized(value) for value in observations if value.timestamp < start]
                after = [_normalized(value) for value in observations if value.timestamp > end]
                immediate_outside = before[-1:] + after[:1]
                contrast = _mean(inside) - _mean(immediate_outside)
                score = pending.confidence + bracket.confidence + contrast
                pairs.append(replace(provisional, score=score))
            pending = None
    return tuple(pairs)


def select_episode_pairs(pairs: Sequence[EpisodePair], cardinality: str) -> tuple[EpisodePair, ...]:
    maximum = MAXIMUM_PAIRS if cardinality == "multi" else 1
    selected = []
    for pair in sorted(pairs, key=lambda value: (-value.score, value.pair_id)):
        start, end = pair.interval
        if all(end <= prior.interval[0] or start >= prior.interval[1] for prior in selected):
            selected.append(pair)
        if len(selected) == maximum:
            break
    return tuple(sorted(selected, key=lambda value: value.interval))


def _normalized(observation: CoarseObservation) -> float:
    return float(observation.normalized) if observation.normalized is not None else 0.0


def _magnitude(observation: CoarseObservation) -> float:
    return abs(_normalized(observation))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _supplementary_timestamps(
    duration: float, pairs: Sequence[EpisodePair], count: int
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if count <= 0:
        return (), ()
    if not pairs or count < 4:
        values = uniform_timestamps(0.0, duration, count)
        return values, ("global_anchor",) * len(values)
    global_count = max(2, count // 4)
    interior_count = count - global_count
    allocations = [interior_count // len(pairs)] * len(pairs)
    for index in range(interior_count % len(pairs)):
        allocations[index] += 1
    values: list[tuple[float, str]] = [
        (timestamp, "global_anchor") for timestamp in uniform_timestamps(0.0, duration, global_count)
    ]
    for pair, allocation in zip(pairs, allocations):
        start, end = pair.interval
        values.extend((timestamp, "interior") for timestamp in uniform_timestamps(start, end, allocation))
    values.sort(key=lambda value: value[0])
    return tuple(value[0] for value in values), tuple(value[1] for value in values)


def _evidence_roles(evidence: TemporalEvidence, timestamps: Sequence[float], roles: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        roles[min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - value))]
        for value in evidence.timestamps
    )


def _nearest_observation(observations: Sequence[CoarseObservation], timestamp: float) -> CoarseObservation:
    return min(observations, key=lambda value: (abs(value.timestamp - timestamp), value.timestamp))


def _role_for_timestamp(
    timestamp: float, corridors: Sequence[BoundaryCorridor], pairs: Sequence[EpisodePair]
) -> str:
    resolved = sorted(
        (value for value in corridors if value.status == "resolved"),
        key=lambda value: (value.role != "start", value.start, value.end),
    )
    for corridor in resolved:
        if corridor.start <= timestamp <= corridor.end:
            return corridor.role
    for corridor in corridors:
        if corridor.status != "resolved" and corridor.start <= timestamp <= corridor.end:
            return "ambiguous_boundary"
    if any(start <= timestamp <= end for start, end in (pair.interval for pair in pairs)):
        return "interior"
    return "global_anchor"


class BoundaryGuidedSparsification(Method):
    name = "boundary-guided-sparsification"

    def __init__(self, retention_ratio: float = RETENTION_RATIO) -> None:
        if not 0 < retention_ratio <= 1:
            raise ValueError("retention ratio must be in (0, 1]")
        self.retention_ratio = retention_ratio

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        self.validate_model(model)
        even_frames = requires_even_frames(model)
        ledger = BudgetLedger(duration_budget(sample.duration))
        evidence_blocks: list[TemporalEvidence] = []
        encoder_calls = 0
        query_scoring_calls = 0

        logical_scout, _ = scout_timestamps(sample.duration)
        scout_times, _, scout_duplicates = duplicate_tubelets(
            logical_scout,
            ("global_anchor",) * len(logical_scout),
            even_frames,
        )
        ledger.reserve(scout_times, tubelet_duplicates=scout_duplicates)
        scout = model.encode(sample, scout_times)
        encoder_calls += 1
        scout.roles = ("global_anchor",) * scout.size
        scout_scores = model.query_scores(scout, sample.query)
        query_scoring_calls += 1
        coarse_raw = aggregate_query_presence(scout, scout_scores)
        if even_frames and len(coarse_raw) != len(logical_scout):
            raise AssertionError(
                f"Qwen produced {len(coarse_raw)} scout units for "
                f"{len(logical_scout)} logical observations"
            )
        coarse, calibration = normalize_presence(coarse_raw)
        scout.metadata["logical_timestamps"] = [float(value) for value in logical_scout]
        scout.metadata["physical_timestamps"] = [float(value) for value in scout_times]
        scout.metadata["effective_temporal_units"] = len(coarse_raw)
        scout.metadata["query_presence"] = [asdict(value) for value in coarse]
        evidence_blocks.append(scout)

        brackets, state_runs = detect_boundaries(coarse, sample.duration)
        candidates = pair_boundaries(brackets, coarse, sample.duration)
        selected = list(select_episode_pairs(candidates, sample.cardinality))
        pair_scores = {value.pair_id: value.score for value in selected}
        refinement_log: list[dict[str, object]] = []
        dropped_pair_ids: set[int] = set()

        def observe(
            times: Sequence[float],
            roles: Sequence[str],
            stage: str,
            *,
            preserve_logical_observations: bool = False,
        ) -> tuple[CoarseObservation, ...]:
            nonlocal encoder_calls, query_scoring_calls
            if preserve_logical_observations:
                physical_times, physical_roles, tubelet_duplicates = duplicate_tubelets(times, roles, even_frames)
                padding = 0
            else:
                physical_times, padding = pad_even(times, even_frames)
                physical_roles = tuple(roles) + ((roles[-1],) if padding else ())
                tubelet_duplicates = 0
            if len(physical_times) > ledger.remaining_frames:
                raise AssertionError(
                    f"{stage} requests {len(physical_times)} frames with {ledger.remaining_frames} remaining"
                )
            ledger.reserve(physical_times, padding, tubelet_duplicates)
            evidence = model.encode(sample, physical_times)
            encoder_calls += 1
            evidence.roles = _evidence_roles(evidence, physical_times, physical_roles)
            scores = model.query_scores(evidence, sample.query)
            query_scoring_calls += 1
            raw = aggregate_query_presence(evidence, scores)
            if preserve_logical_observations and even_frames and len(raw) != len(times):
                raise AssertionError(
                    f"Qwen temporal collapse in {stage}: "
                    f"{len(raw)} units for {len(times)} logical observations"
                )
            observations = []
            for value in raw:
                normalized, state = state_for_score(value.aggregate, calibration)
                observations.append(replace(value, normalized=normalized, state=state))
            observations = tuple(observations)
            evidence.metadata["query_presence"] = [asdict(value) for value in observations]
            evidence_blocks.append(evidence)
            refinement_log.append(
                {
                    "stage": stage,
                    "requested_timestamps": [float(value) for value in times],
                    "physical_timestamps": [float(value) for value in physical_times],
                    "padding_frames": padding,
                    "tubelet_duplicate_frames": tubelet_duplicates,
                    "effective_temporal_units": len(raw),
                    "observations": [asdict(value) for value in observations],
                }
            )
            return observations

        corridors = self._refine(
            selected,
            pair_scores,
            ledger,
            even_frames,
            observe,
            dropped_pair_ids,
        )
        selected = [value for value in selected if value.pair_id not in dropped_pair_ids]
        selected_ids = {value.pair_id for value in selected}
        corridors = [value for value in corridors if value.pair_id in selected_ids]

        fallback_used = not selected
        supplement_times, supplement_roles = _supplementary_timestamps(
            sample.duration, selected, ledger.remaining_frames
        )
        if supplement_times:
            while supplement_times and len(pad_even(supplement_times, even_frames)[0]) > ledger.remaining_frames:
                supplement_times = supplement_times[:-1]
                supplement_roles = supplement_roles[:-1]
            if supplement_times:
                observe(supplement_times, supplement_roles, "fallback-anchors" if fallback_used else "final-evidence")

        if ledger.requested_frames > ledger.budget:
            raise AssertionError(f"BGS exceeded sampled-frame budget {ledger.budget}")

        merged = TemporalEvidence.concatenate(evidence_blocks)
        merged.roles = tuple(_role_for_timestamp(value, corridors, selected) for value in merged.timestamps)
        merged_scores = model.query_scores(merged, sample.query)
        query_scoring_calls += 1
        compact, protected, role_losses = self._pack(merged, merged_scores, corridors, pair_scores, model)
        prediction = model.predict(sample, compact, GroundingContext(0.0, sample.duration))
        role_counts = {role: compact.roles.count(role) for role in sorted(set(compact.roles))}
        constants = {
            "cell_seconds": CELL_SECONDS,
            "persistence": PERSISTENCE,
            "absent_threshold": ABSENT_THRESHOLD,
            "present_threshold": PRESENT_THRESHOLD,
            "maximum_pairs": MAXIMUM_PAIRS,
            "target_resolution": TARGET_RESOLUTION,
            "ambiguity_probe_rounds": AMBIGUITY_PROBE_ROUNDS,
            "retention_ratio": self.retention_ratio,
        }
        return Prediction(
            prediction.spans,
            prediction.raw_output,
            {
                **prediction.telemetry,
                **ledger.to_dict(),
                "sampled_frames": ledger.requested_frames,
                "coarse_cells": len(logical_scout),
                "scout_logical_frames": len(logical_scout),
                "scout_physical_frames": len(scout_times),
                "coarse_observations": [asdict(value) for value in coarse],
                "presence_calibration": asdict(calibration),
                "state_runs": [asdict(value) for value in state_runs],
                "brackets": [asdict(value) for value in brackets],
                "candidate_pairs": [asdict(value) for value in candidates],
                "selected_pair_ids": [value.pair_id for value in selected],
                "dropped_pair_ids": sorted(dropped_pair_ids),
                "refinement": refinement_log,
                "boundary_corridors": [asdict(value) for value in corridors],
                "selected_roles": role_counts,
                "protected_anchors": len(protected),
                "cap_induced_role_losses": role_losses,
                "encoder_calls": encoder_calls,
                "query_scoring_calls": query_scoring_calls,
                "llm_or_fusion_calls": 1,
                "created_evidence": merged.size,
                "retained_evidence": compact.size,
                "retention_ratio": compact.size / merged.size,
                "fallback_used": fallback_used,
                "policy": "boundary-guided-sparsification",
                "constants": constants,
                "qwen_tubelet_duplication": even_frames,
                "absolute_timestamps_preserved": True,
            },
        )

    def _refine(
        self,
        pairs: Sequence[EpisodePair],
        pair_scores: dict[int, float],
        ledger: BudgetLedger,
        even_frames: bool,
        observe,
        dropped_pair_ids: set[int],
    ) -> list[BoundaryCorridor]:
        corridors: list[BoundaryCorridor] = []
        active: list[_ActiveBoundary] = []
        for pair in pairs:
            for bracket in (pair.start, pair.end):
                if bracket.edge:
                    corridors.append(
                        BoundaryCorridor(
                            bracket.start,
                            bracket.end,
                            bracket.role,
                            "resolved",
                            "video_edge",
                            bracket.confidence,
                            pair.pair_id,
                        )
                    )
                    continue
                width = bracket.end - bracket.start
                active.append(
                    _ActiveBoundary(
                        pair.pair_id,
                        bracket.role,
                        bracket.start,
                        bracket.end,
                        bracket.left_state,
                        bracket.right_state,
                        bracket.confidence,
                        max(0, math.ceil(math.log2(width / TARGET_RESOLUTION))) if width > TARGET_RESOLUTION else 0,
                    )
                )

        while active:
            unfinished = [
                value
                for value in active
                if value.right - value.left > TARGET_RESOLUTION and value.depth < value.maximum_depth
            ]
            for value in active:
                if value not in unfinished:
                    corridors.append(self._finished_corridor(value))
            active = unfinished
            if not active:
                break
            active = self._fit_batch(
                active,
                ledger.remaining_frames,
                even_frames,
                pair_scores,
                dropped_pair_ids,
                frames_per_item=2 if even_frames else 1,
            )
            if not active:
                break
            midpoints = [(value.left + value.right) / 2.0 for value in active]
            observations = observe(
                midpoints,
                [value.role for value in active],
                "midpoint",
                preserve_logical_observations=True,
            )
            next_active: list[_ActiveBoundary] = []
            uncertain: list[_ActiveBoundary] = []
            for value, midpoint in zip(active, midpoints):
                state = _nearest_observation(observations, midpoint).state
                value.depth += 1
                if state == value.left_state:
                    value.left = midpoint
                    next_active.append(value)
                elif state == value.right_state:
                    value.right = midpoint
                    next_active.append(value)
                else:
                    uncertain.append(value)

            probeable = [value for value in uncertain if value.ambiguity_probes < AMBIGUITY_PROBE_ROUNDS]
            unresolved = [value for value in uncertain if value not in probeable]
            for value in unresolved:
                corridors.append(self._ambiguous_corridor(value, "persistent_uncertainty"))
            if probeable:
                probeable = self._fit_batch(
                    probeable,
                    ledger.remaining_frames,
                    even_frames,
                    pair_scores,
                    dropped_pair_ids,
                    frames_per_item=4 if even_frames else 2,
                )
                probe_times = [
                    timestamp
                    for value in probeable
                    for timestamp in (
                        value.left + (value.right - value.left) / 4.0,
                        value.left + 3.0 * (value.right - value.left) / 4.0,
                    )
                ]
                if probe_times:
                    probe_roles = [value.role for value in probeable for _ in range(2)]
                    probe_observations = observe(
                        probe_times,
                        probe_roles,
                        "ambiguity-probe",
                        preserve_logical_observations=True,
                    )
                    for index, value in enumerate(probeable):
                        left_probe, right_probe = probe_times[2 * index : 2 * index + 2]
                        left_state = _nearest_observation(probe_observations, left_probe).state
                        right_state = _nearest_observation(probe_observations, right_probe).state
                        value.ambiguity_probes += 1
                        if left_state == value.left_state and right_state == value.right_state:
                            value.left, value.right = left_probe, right_probe
                            next_active.append(value)
                        else:
                            corridors.append(self._ambiguous_corridor(value, "ambiguous_quarter_probes"))
            active = [value for value in next_active if value.pair_id not in dropped_pair_ids]
        return corridors

    @staticmethod
    def _fit_batch(
        values: Sequence[_ActiveBoundary],
        available: int,
        even_frames: bool,
        pair_scores: dict[int, float],
        dropped_pair_ids: set[int],
        frames_per_item: int = 1,
    ) -> list[_ActiveBoundary]:
        kept = list(values)
        while kept:
            frame_count = len(kept) * frames_per_item
            if even_frames and frame_count % 2:
                frame_count += 1
            if frame_count <= available:
                break
            drop = min({value.pair_id for value in kept}, key=lambda pair_id: (pair_scores[pair_id], -pair_id))
            dropped_pair_ids.add(drop)
            kept = [value for value in kept if value.pair_id != drop]
        return kept

    @staticmethod
    def _finished_corridor(value: _ActiveBoundary) -> BoundaryCorridor:
        resolved = value.right - value.left <= TARGET_RESOLUTION
        return BoundaryCorridor(
            value.left,
            value.right,
            value.role,
            "resolved" if resolved else "ambiguous",
            "target_resolution" if resolved else "depth_exhausted",
            value.confidence,
            value.pair_id,
        )

    @staticmethod
    def _ambiguous_corridor(value: _ActiveBoundary, reason: str) -> BoundaryCorridor:
        return BoundaryCorridor(
            value.left,
            value.right,
            value.role,
            "ambiguous",
            reason,
            value.confidence,
            value.pair_id,
        )

    def _pack(self, evidence, scores, corridors, pair_scores, model):
        cap = model.maximum_evidence_units or evidence.size
        base_target = min(cap, max(1, round(evidence.size * self.retention_ratio)))
        protected: set[int] = set()
        ranked_corridors = sorted(
            corridors,
            key=lambda value: (-pair_scores.get(value.pair_id, float("-inf")), value.role, value.start),
        )
        role_losses = []
        for corridor_index, corridor in enumerate(ranked_corridors):
            if len(protected) == cap:
                role_losses.extend(
                    {"pair_id": value.pair_id, "role": value.role}
                    for value in ranked_corridors[corridor_index:]
                )
                break
            candidates = [
                index
                for index, timestamp in enumerate(evidence.timestamps)
                if corridor.start <= timestamp <= corridor.end
            ]
            if not candidates:
                center = (corridor.start + corridor.end) / 2.0
                candidates = [min(range(evidence.size), key=lambda index: abs(evidence.timestamps[index] - center))]
            protected.add(max(candidates, key=lambda index: (float(scores[index]), -index)))
        target = min(cap, max(base_target, len(protected)))
        coverage_count = min(target, max(2, math.ceil(math.sqrt(target))))
        coverage = sorted(
            temporal_anchor_indices(evidence, scores, coverage_count),
            key=lambda index: (evidence.timestamps[index], index),
        )
        for index in coverage:
            if len(protected) == target:
                break
            protected.add(index)
        return pack_evidence(evidence, scores, target, protected), protected, role_losses
