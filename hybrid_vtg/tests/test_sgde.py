"""Unit and integration tests for SGDE-64 (Idea 3)."""

from pathlib import Path

import numpy as np
import torch

from hybrid_vtg.contracts import (
    GroundingContext,
    ModelBackend,
    Prediction,
    Sample,
    ScoredSpan,
    TemporalEvidence,
)
from hybrid_vtg.methods.sgde_64 import (
    SGDE64,
    CandidateProposal,
    ScoutProvider,
    ScoutTimeline,
    extract_candidate_proposals,
    plan_sgde_evidence,
)
from hybrid_vtg.methods.sgde_64.planning import FRAME_BUDGET
from hybrid_vtg.methods.sgde_64.proposals import (
    extract_hysteresis_components,
    extract_multiscale_density_windows,
    extract_penalized_intervals,
    temporal_nms,
)
from hybrid_vtg.methods.sgde_64.scout import normalize_timeline, smooth_timeline


def test_smooth_and_normalize_timeline():
    raw = np.array([0.1, 0.1, 0.9, 0.1, 0.1], dtype=np.float32)
    smoothed = smooth_timeline(raw, window_size=3)
    assert len(smoothed) == len(raw)
    assert smoothed[2] < 0.9  # peak softened by neighbors
    assert smoothed[1] > 0.1  # neighbor elevated

    z, med, mad = normalize_timeline(smoothed)
    assert len(z) == len(raw)
    assert mad > 0
    assert float(np.argmax(z)) == 2


def test_hysteresis_components():
    timestamps = np.linspace(0, 50, 51).astype(np.float32)
    z_scores = np.zeros(51, dtype=np.float32)
    # Event 1: strong peak [10, 15]
    z_scores[10:16] = [0.4, 0.8, 1.5, 1.2, 0.5, 0.35]
    # Noise: weak bump [30, 32] (does not reach high_thresh 1.0)
    z_scores[30:33] = [0.4, 0.6, 0.4]

    proposals = extract_hysteresis_components(timestamps, z_scores, high_threshold=1.0, low_threshold=0.3)
    assert len(proposals) == 1
    assert 9.0 <= proposals[0].start <= 11.0
    assert 14.0 <= proposals[0].end <= 16.0
    assert proposals[0].peak_z >= 1.5


def test_penalized_intervals():
    timestamps = np.linspace(0, 50, 51).astype(np.float32)
    z_scores = np.zeros(51, dtype=np.float32)
    z_scores[20:26] = [0.8, 1.4, 2.0, 1.6, 0.9, 0.6]

    proposals = extract_penalized_intervals(timestamps, z_scores, tau=0.5, lambda_len=0.05)
    assert len(proposals) >= 1
    best = max(proposals, key=lambda p: p.penalized_score)
    assert 19.0 <= best.start <= 21.0
    assert 24.0 <= best.end <= 26.0
    assert best.penalized_score > 0


def test_multiscale_density_windows():
    timestamps = np.linspace(0, 100, 101).astype(np.float32)
    z_scores = np.zeros(101, dtype=np.float32)
    z_scores[40:45] = 2.0

    proposals = extract_multiscale_density_windows(timestamps, z_scores, 100.0, scales=(5.0, 10.0, 20.0))
    assert len(proposals) >= 1
    assert any(35.0 <= p.start <= 43.0 for p in proposals)


def test_temporal_nms():
    cands = [
        CandidateProposal(10.0, 20.0, peak_z=2.0, mean_z=1.5, penalized_score=5.0, score=1.8, source="a"),
        CandidateProposal(11.0, 21.0, peak_z=1.9, mean_z=1.4, penalized_score=4.5, score=1.6, source="b"),  # overlaps
        CandidateProposal(40.0, 50.0, peak_z=1.5, mean_z=1.2, penalized_score=3.0, score=1.3, source="c"),
    ]
    kept = temporal_nms(cands, iou_threshold=0.5, max_candidates=4)
    assert len(kept) == 2
    assert kept[0].start == 10.0
    assert kept[1].start == 40.0


def test_extract_candidate_proposals_confidence():
    timestamps = np.linspace(0, 100, 101).astype(np.float32)
    flat_scores = np.ones(101, dtype=np.float32) * 0.1
    flat_timeline = ScoutTimeline(
        timestamps=timestamps,
        raw_scores=flat_scores,
        smoothed_scores=flat_scores,
        z_scores=np.zeros(101, dtype=np.float32),
        median=0.1,
        mad=0.0,
        peak_z=0.0,
        model_id="test",
        cached=True,
    )
    cands, confident = extract_candidate_proposals(flat_timeline, 100.0)
    assert not confident
    assert len(cands) == 0

    # Confident timeline
    z_scores = np.zeros(101, dtype=np.float32)
    z_scores[50:56] = [0.5, 1.2, 2.5, 1.8, 0.7, 0.4]
    good_timeline = ScoutTimeline(
        timestamps=timestamps,
        raw_scores=z_scores,
        smoothed_scores=z_scores,
        z_scores=z_scores,
        median=0.0,
        mad=0.2,
        peak_z=2.5,
        model_id="test",
        cached=True,
    )
    cands, confident = extract_candidate_proposals(good_timeline, 100.0, cardinality="single")
    assert confident
    assert len(cands) == 1
    assert 48.0 <= cands[0].start <= 52.0


def test_plan_sgde_evidence_budget_and_roles():
    candidates = [
        CandidateProposal(20.0, 35.0, 2.0, 1.5, 5.0, 1.8, "test"),
        CandidateProposal(60.0, 75.0, 1.8, 1.3, 4.0, 1.5, "test"),
    ]
    observations = plan_sgde_evidence(candidates, 100.0, budget=FRAME_BUDGET)

    assert len(observations) == FRAME_BUDGET
    # Strictly sorted
    times = [o.timestamp for o in observations]
    assert times == sorted(times)

    roles = {o.role for o in observations}
    assert "global_anchor" in roles
    assert "candidate" in roles
    assert "boundary_transition" in roles


def test_plan_sgde_evidence_empty_fallback():
    observations = plan_sgde_evidence([], 120.0, budget=FRAME_BUDGET)
    assert len(observations) == FRAME_BUDGET
    assert all(o.role == "exploration" for o in observations)


class _MockModel(ModelBackend):
    name = "mock-grounder"

    def __init__(self):
        self.encoder_pruning = "none"
        self.post_pruning = "none"
        self.encode_calls = []
        self.predict_calls = []

    def encode(self, sample, timestamps):
        self.encode_calls.append(tuple(timestamps))
        return TemporalEvidence(
            torch.ones((len(timestamps), 4)),
            tuple(timestamps),
            len(timestamps),
            metadata={"dense_evidence_units": len(timestamps) * 4},
        )

    def query_scores(self, evidence, query):
        del query
        return torch.ones(evidence.size)

    def predict(self, sample, evidence, context: GroundingContext):
        self.predict_calls.append((sample, evidence, context))
        return Prediction((ScoredSpan(22.0, 34.0, 0.95),), "[[22.0, 34.0]]")


class _MockScoutProvider(ScoutProvider):
    def __init__(self, timeline: ScoutTimeline):
        super().__init__()
        self._mock_timeline = timeline

    def compute_timeline(self, sample: Sample, cache_root: Path, *, smoothing_window: int = 3):
        return self._mock_timeline


def test_sgde_64_end_to_end(tmp_path: Path):
    timestamps = np.linspace(0, 100, 101).astype(np.float32)
    z_scores = np.zeros(101, dtype=np.float32)
    z_scores[20:30] = np.linspace(0.5, 2.5, 10)

    timeline = ScoutTimeline(
        timestamps=timestamps,
        raw_scores=z_scores,
        smoothed_scores=z_scores,
        z_scores=z_scores,
        median=0.0,
        mad=0.2,
        peak_z=2.5,
        model_id="test-scout",
        cached=True,
    )

    scout = _MockScoutProvider(timeline)
    method = SGDE64(scout_provider=scout)
    model = _MockModel()

    sample = Sample(
        id="test::0",
        video="test_vid",
        video_path=Path(__file__),
        duration=100.0,
        query="a person rides a bike",
        cardinality="single",
    )

    pred = method.run(sample, model, tmp_path)

    assert len(pred.spans) == 1
    assert pred.spans[0].start == 22.0
    assert pred.spans[0].end == 34.0
    assert pred.telemetry["method"] == "sgde-64"
    assert pred.telemetry["route_mode"] == "scout-guided"
    assert pred.telemetry["scout_confident"] is True
    assert pred.telemetry["total_frames"] == FRAME_BUDGET
    assert pred.telemetry["encoder_calls"] == 1
    assert pred.telemetry["primary_grounder_calls"] == 1
    assert len(model.encode_calls) == 1
    assert len(model.predict_calls) == 1


def test_sgde_64_fallback_mode(tmp_path: Path):
    timestamps = np.linspace(0, 100, 101).astype(np.float32)
    flat_scores = np.ones(101, dtype=np.float32) * 0.1

    timeline = ScoutTimeline(
        timestamps=timestamps,
        raw_scores=flat_scores,
        smoothed_scores=flat_scores,
        z_scores=np.zeros(101, dtype=np.float32),
        median=0.1,
        mad=0.0,
        peak_z=0.0,
        model_id="test-scout",
        cached=True,
    )

    scout = _MockScoutProvider(timeline)
    method = SGDE64(scout_provider=scout)
    model = _MockModel()

    sample = Sample(
        id="test::0",
        video="test_vid",
        video_path=Path(__file__),
        duration=100.0,
        query="something obscure",
        cardinality="single",
    )

    pred = method.run(sample, model, tmp_path)
    assert pred.telemetry["route_mode"] == "full-video-fallback"
    assert pred.telemetry["scout_confident"] is False
    assert pred.telemetry["total_frames"] == FRAME_BUDGET
    assert pred.telemetry["observation_role_counts"]["exploration"] == FRAME_BUDGET


def test_sgde_64_multi_candidate(tmp_path: Path):
    timestamps = np.linspace(0, 100, 101).astype(np.float32)
    z_scores = np.zeros(101, dtype=np.float32)
    z_scores[10:18] = np.linspace(0.8, 2.2, 8)
    z_scores[60:68] = np.linspace(0.8, 2.0, 8)

    timeline = ScoutTimeline(
        timestamps=timestamps,
        raw_scores=z_scores,
        smoothed_scores=z_scores,
        z_scores=z_scores,
        median=0.0,
        mad=0.2,
        peak_z=2.2,
        model_id="test-scout",
        cached=True,
    )

    scout = _MockScoutProvider(timeline)
    method = SGDE64(scout_provider=scout)
    model = _MockModel()

    sample = Sample(
        id="test::1",
        video="test_vid",
        video_path=Path(__file__),
        duration=100.0,
        query="multiple actions occurring",
        cardinality="multi",
    )

    pred = method.run(sample, model, tmp_path)
    assert pred.telemetry["route_mode"] == "scout-guided"
    assert len(pred.telemetry["candidates"]) >= 2
    assert pred.telemetry["total_frames"] == FRAME_BUDGET


def test_scout_provider_cache_discovery(tmp_path: Path):
    feature_dir = tmp_path / "custom_features" / "google--siglip2-base-patch16-224" / "omtg"
    feature_dir.mkdir(parents=True)
    video_dir = feature_dir / "video_embeddings"
    video_dir.mkdir(parents=True)

    # Save mock video embedding
    timestamps = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    embeddings = np.ones((3, 64), dtype=np.float16)
    np.savez_compressed(video_dir / "sample_vid.npz", timestamps=timestamps, embeddings=embeddings)

    # Save mock query embedding
    np.savez_compressed(
        feature_dir / "queries.npz",
        ids=np.array(["sample_id::0"]),
        embeddings=np.ones((1, 64), dtype=np.float16),
    )

    provider = ScoutProvider(
        model_id="google/siglip2-base-patch16-224",
        feature_roots=(tmp_path / "custom_features",),
    )

    sample = Sample(
        id="sample_id::0",
        video="sample_vid",
        video_path=Path(__file__),
        duration=3.0,
        query="mock query",
        group="omtg",
    )

    timeline = provider.compute_timeline(sample, tmp_path / "cache")
    assert timeline.cached is True
    assert len(timeline.timestamps) == 3
    assert timeline.raw_scores.shape == (3,)
