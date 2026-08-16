from pathlib import Path

import torch

from hybrid_vtg.contracts import (
    GroundingContext,
    ModelBackend,
    Prediction,
    Sample,
    ScoredSpan,
    TemporalEvidence,
)
from hybrid_vtg.methods.anchored_corridor_64 import (
    FRAME_BUDGET,
    AnchoredCorridor64,
    plan_anchored_evidence,
    route_confidence,
)
from hybrid_vtg.methods.coarse_to_fine_64 import EmbeddingRouter, Window


def test_anchored_plan_is_exact_and_protects_every_routed_window():
    windows = [Window(index * 20.0, (index + 1) * 20.0) for index in range(5)]
    observations = plan_anchored_evidence(
        windows,
        [Window(36.0, 64.0)],
        100.0,
    )

    assert len(observations) == FRAME_BUDGET
    assert [value.timestamp for value in observations] == sorted(
        value.timestamp for value in observations
    )
    anchors = [value for value in observations if value.role == "global_anchor"]
    assert len(anchors) == len(windows)
    assert {value.region_id for value in anchors} == set(range(len(windows)))
    assert sum(value.role == "corridor" for value in observations) > len(anchors)


def test_route_confidence_fails_open_on_query_view_disagreement():
    decision = route_confidence(
        [0.9, 0.1, 0.0],
        {
            "query_view_scores": {"raw": [0.1, 0.9, 0.0]},
            "window_similarity": [0.9, 0.1, 0.0],
            "frame_occurrence_similarity": [0.9, 0.1, 0.0],
        },
    )

    assert not decision.confident
    assert "query-view-disagreement" in decision.reasons


class _StaticMultiRouter:
    def __init__(self, scores, telemetry):
        self.scores = list(scores)
        self.last_telemetry = telemetry
        self.calls = []

    def rank_multigranular(self, sample, windows, frames_per_window, cache_root):
        self.calls.append((sample, tuple(windows), frames_per_window, cache_root))
        return list(self.scores)


class _OneCallBackend(ModelBackend):
    name = "one-call"

    def __init__(self):
        self.encoder_pruning = "none"
        self.post_pruning = "none"
        self.encode_calls = []
        self.predict_calls = []

    def encode(self, sample, timestamps):
        self.encode_calls.append(tuple(timestamps))
        return TemporalEvidence(
            torch.ones((len(timestamps), 2)),
            tuple(timestamps),
            len(timestamps),
            metadata={"dense_evidence_units": len(timestamps) * 2},
        )

    def query_scores(self, evidence, query):
        del query
        return torch.ones(evidence.size)

    def predict(self, sample, evidence, context: GroundingContext):
        self.predict_calls.append((sample, evidence, context))
        return Prediction((ScoredSpan(10.0, 20.0),), "[[10, 20]]")


def _sample():
    return Sample(
        "sample",
        "video",
        Path(__file__),
        120.0,
        "person opens a container",
        cardinality="multi",
    )


def _confident_telemetry():
    return {
        "query_view_scores": {"raw": [0.95, 0.2, 0.1]},
        "window_similarity": [0.9, 0.2, 0.1],
        "frame_occurrence_similarity": [0.85, 0.3, 0.0],
    }


def test_anchored_method_uses_one_encode_and_one_grounding_call(monkeypatch, tmp_path):
    router = _StaticMultiRouter([0.95, 0.2, 0.1], _confident_telemetry())
    method = AnchoredCorridor64(router=router)
    windows = [Window(0.0, 40.0), Window(38.0, 80.0), Window(78.0, 120.0)]
    monkeypatch.setattr(method, "_cached_windows", lambda sample, cache: (windows, "content"))
    backend = _OneCallBackend()

    result = method.run(_sample(), backend, tmp_path)

    assert len(backend.encode_calls) == len(backend.predict_calls) == 1
    assert len(backend.encode_calls[0]) == FRAME_BUDGET
    assert backend.predict_calls[0][2] == GroundingContext(0.0, 120.0)
    assert result.telemetry["route_mode"] == "corridor"
    assert result.telemetry["encoder_calls"] == 1
    assert result.telemetry["primary_grounder_calls"] == 1
    assert result.telemetry["observation_role_counts"]["global_anchor"] == len(windows)
    assert result.telemetry["observation_role_counts"]["corridor"] > 0
    assert set(backend.predict_calls[0][1].roles) >= {"global_anchor", "corridor"}


def test_uncertain_route_falls_back_to_uniform_full_video(monkeypatch, tmp_path):
    telemetry = {
        "query_view_scores": {"raw": [0.1, 0.9, 0.0]},
        "window_similarity": [0.9, 0.1, 0.0],
        "frame_occurrence_similarity": [0.9, 0.1, 0.0],
    }
    method = AnchoredCorridor64(router=_StaticMultiRouter([0.9, 0.1, 0.0], telemetry))
    windows = [Window(0.0, 40.0), Window(38.0, 80.0), Window(78.0, 120.0)]
    monkeypatch.setattr(method, "_cached_windows", lambda sample, cache: (windows, "content"))
    backend = _OneCallBackend()

    result = method.run(_sample(), backend, tmp_path)

    assert result.telemetry["route_mode"] == "full-video-fallback"
    assert result.telemetry["selected"] == []
    assert result.telemetry["corridors"] == []
    assert result.telemetry["observation_role_counts"] == {
        "global_anchor": FRAME_BUDGET,
        "corridor": 0,
        "exploration": 0,
    }
    timestamps = backend.encode_calls[0]
    assert len(timestamps) == FRAME_BUDGET
    assert timestamps[0] < 1.0 and timestamps[-1] > 119.0


def test_embedding_router_combines_four_granularity_views(monkeypatch, tmp_path):
    class FakeEmbeddingModel:
        def encode(self, values, **kwargs):
            del kwargs
            if "text" in values[0]:
                instruction = values[0]["instruction"]
                if "high-level" in instruction:
                    row = [0.0, 1.0]
                elif "action sequence" in instruction:
                    row = [0.5, 0.5]
                else:
                    row = [1.0, 0.0]
                return torch.tensor([row for _ in values])
            rows = []
            for value in values:
                path = value.get("image") or value["video"][0]
                rows.append([1.0, 0.0] if "window-0" in path else [0.0, 1.0])
            return torch.tensor(rows)

    def fake_extract(video, timestamps, destination):
        del video
        return tuple(destination / f"{index}.jpg" for index, _ in enumerate(timestamps))

    monkeypatch.setattr("hybrid_vtg.methods.coarse_to_fine_64.extract_frames", fake_extract)
    router = EmbeddingRouter()
    router._model = FakeEmbeddingModel()
    video = tmp_path / "video.mp4"
    video.touch()
    sample = Sample("1", "video", video, 20.0, "open the jar")

    scores = router.rank_multigranular(
        sample,
        [Window(0.0, 10.0), Window(10.0, 20.0)],
        2,
        tmp_path,
    )

    assert scores == [1.0, 0.5]
    assert router.last_telemetry["query_views"] == ["raw", "coarse", "actions", "details"]
    assert set(router.last_telemetry["query_view_scores"]) == {
        "raw",
        "coarse",
        "actions",
        "details",
    }
