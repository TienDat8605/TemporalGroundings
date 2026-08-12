import math
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
from hybrid_vtg.methods.coarse_to_fine_64 import (
    FRAME_BUDGET,
    CoarseToFine64,
    EmbeddingRouter,
    Window,
    WindowCandidate,
    content_windows,
    distribute_frames,
    fuse_cross_window_spans,
    strict_budget,
    uniform_windows,
)
from hybrid_vtg.models.qwen import QwenEvidenceBackend


def test_coarse_to_fine_policy_never_exceeds_64_frames():
    windows = uniform_windows(900.0)
    routed, policy = strict_budget(windows)
    allocations = distribute_frames(policy["local_budget"], policy["selected_windows"])
    assert len(routed) <= len(windows)
    assert policy["router_frames"] + sum(allocations) == FRAME_BUDGET
    assert all(value >= 2 and value % 2 == 0 for value in allocations)


def test_single_window_is_a_64_frame_bypass():
    routed, policy = strict_budget([Window(0.0, 10.0)])
    assert routed == [Window(0.0, 10.0)]
    assert policy["router_frames"] + policy["local_budget"] == FRAME_BUDGET


def test_coarse_to_fine_requests_headless_opencv_scene_backend(monkeypatch):
    import scenedetect

    received = {}

    def fake_open_video(path, *, backend):
        received.update(path=path, backend=backend)
        raise RuntimeError("stop after backend selection")

    monkeypatch.setattr(scenedetect, "open_video", fake_open_video)
    windows, policy = content_windows(Path("video.mp4"), 90.0)
    assert received == {"path": "video.mp4", "backend": "opencv"}
    assert windows == uniform_windows(90.0)
    assert policy == "uniform-fallback"


def test_coarse_to_fine_prepare_caches_windows_for_all_samples(monkeypatch, tmp_path):
    import json

    calls = []

    def fake_content_windows(video_path, duration):
        calls.append((video_path, duration))
        return [Window(0.0, duration)], "content"

    monkeypatch.setattr(
        "hybrid_vtg.methods.coarse_to_fine_64.content_windows",
        fake_content_windows,
    )
    method = CoarseToFine64()
    samples = [
        Sample("a", "va", tmp_path / "va.mp4", 20.0, "q a"),
        Sample("b", "vb", tmp_path / "vb.mp4", 30.0, "q b"),
    ]
    cache_root = tmp_path / "prepare"
    method.prepare(samples, cache_root)

    assert len(calls) == 2
    assert (cache_root / "a.json").is_file()
    assert (cache_root / "b.json").is_file()
    value = json.loads((cache_root / "a.json").read_text())
    assert value["source"] == "content"
    assert value["windows"] == [{"start": 0.0, "end": 20.0}]

    calls.clear()
    method.prepare(samples, cache_root)
    assert calls == []


def test_coarse_to_fine_router_uses_sentence_transformer_encode(monkeypatch, tmp_path):
    calls = []

    class FakeEmbeddingModel:
        def encode(self, values, **kwargs):
            calls.append((values, kwargs))
            if isinstance(values[0], str):
                return torch.tensor([[1.0, 0.0]])
            return torch.tensor([[0.5, 0.5], [1.0, 0.0]])

    monkeypatch.setattr(
        "hybrid_vtg.methods.coarse_to_fine_64.extract_frames",
        lambda video, timestamps, destination: [destination / f"{index}.jpg" for index, _ in enumerate(timestamps)],
    )
    router = EmbeddingRouter()
    router._model = FakeEmbeddingModel()
    sample = Sample("1", "video", tmp_path / "video.mp4", 20.0, "open the door")
    scores = router.rank(sample, [Window(0.0, 10.0), Window(10.0, 20.0)], 2, tmp_path)

    assert scores == [0.5, 1.0]
    assert calls[0][0] == ["open the door"]
    assert [list(value) for value in calls[1][0]] == [["video"], ["video"]]
    assert all(call[1]["normalize_embeddings"] for call in calls)


def test_cross_window_fusion_preserves_same_window_and_adjacent_occurrences():
    candidates = [
        WindowCandidate(0, 2.0, ScoredSpan(10.0, 20.0, 1.0)),
        WindowCandidate(0, 2.0, ScoredSpan(10.5, 19.5, 0.5)),
        WindowCandidate(1, 0.0, ScoredSpan(12.0, 22.0, 0.8)),
        WindowCandidate(2, 1.0, ScoredSpan(22.0, 24.0, 0.7)),
    ]

    spans, groups = fuse_cross_window_spans(candidates, 30.0)

    assert len(spans) == 3
    fused = next(group for group in groups if group["fused"])
    assert {member["window_index"] for member in fused["members"]} == {0, 1}
    assert len({member["window_index"] for member in fused["members"]}) == len(fused["members"])
    assert any(span.start == 10.5 and span.end == 19.5 for span in spans)
    assert any(span.start == 22.0 and span.end == 24.0 for span in spans)
    assert [span.start for span in spans] == sorted(span.start for span in spans)


def test_cross_window_fusion_uses_strict_iou_and_router_softmax_weights():
    candidates = [
        WindowCandidate(0, 2.0, ScoredSpan(10.0, 20.0)),
        WindowCandidate(1, 0.0, ScoredSpan(12.0, 22.0)),
        # Intersection 6, union 10: exactly 0.6 and therefore not a duplicate.
        WindowCandidate(2, 1.0, ScoredSpan(0.0, 8.0)),
        WindowCandidate(3, 1.0, ScoredSpan(2.0, 10.0)),
    ]

    spans, groups = fuse_cross_window_spans(candidates, 30.0)

    expected = (math.exp(2.0) * 10.0 + 12.0) / (math.exp(2.0) + 1.0)
    fused = next(span for span in spans if 10.0 < span.start < 12.0)
    assert math.isclose(fused.start, expected)
    assert fused.start < 11.0
    assert len(groups) == 3
    assert sum(group["fused"] for group in groups) == 1


class _StaticRouter:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rank(self, sample, windows, frames_per_window, cache_dir):
        self.calls.append((sample, tuple(windows), frames_per_window, cache_dir))
        return list(self.scores)


class _PruningTelemetryBackend(ModelBackend):
    name = "fake-qwen"

    def __init__(self, *, encoder_pruning="mage", post_pruning="semvid"):
        self.encoder_pruning = encoder_pruning
        self.post_pruning = post_pruning
        self.encode_calls = []
        self.predict_calls = []

    def encode(self, sample, timestamps):
        self.encode_calls.append((self.encoder_pruning, tuple(timestamps)))
        embeddings = torch.ones((len(timestamps), 2))
        return TemporalEvidence(
            embeddings,
            tuple(timestamps),
            len(timestamps),
            metadata={
                "dense_evidence_units": len(timestamps) * 8,
                "encoder_retained_evidence_units": len(timestamps) * 4,
            },
        )

    def query_scores(self, evidence, query):
        del query
        return torch.ones(evidence.size)

    def predict(self, sample, evidence, context: GroundingContext):
        del sample
        self.predict_calls.append((self.post_pruning, context, evidence.source_frames))
        span = ScoredSpan(41.0, 45.0)
        return Prediction(
            (span,),
            "[[0, 4]]",
            {
                "llm_input_tokens": 100,
                "llm_output_tokens": 7,
                "semvid": {
                    "semvid_input_evidence_units": evidence.size,
                    "semvid_retained_evidence_units": max(1, evidence.size // 2),
                },
            },
        )


def test_local_grounding_preserves_pruning_budget_and_window_telemetry(monkeypatch, tmp_path):
    router = _StaticRouter([1.0, 0.9, 0.1])
    method = CoarseToFine64(router=router)
    monkeypatch.setattr(
        method,
        "_cached_windows",
        lambda sample, cache: ([Window(0.0, 45.0), Window(41.0, 86.0), Window(82.0, 120.0)], "content"),
    )
    backend = _PruningTelemetryBackend()
    sample = Sample("long", "video", Path(__file__), 120.0, "event", cardinality="multi")

    result = method.run(sample, backend, tmp_path)

    assert result.telemetry["total_frames"] == FRAME_BUDGET
    assert len(backend.encode_calls) == len(backend.predict_calls) == 2
    assert all(policy == "mage" for policy, _ in backend.encode_calls)
    assert all(policy == "semvid" for policy, _, _ in backend.predict_calls)
    selected = [window for window in result.telemetry["window_telemetry"] if window["selected"]]
    assert sum(window["allocated_frames"] for window in selected) == result.telemetry["grounder_frames"]
    assert all(window["encoder_dense_tokens"] > window["encoder_retained_tokens"] for window in selected)
    assert all(window["semvid_input_tokens"] > window["semvid_output_tokens"] for window in selected)
    assert all(window["timing"]["total_seconds"] >= 0 for window in selected)
    assert len(result.spans) == 1
    assert result.telemetry["fusion_groups"][0]["fused"] is True


def test_short_video_bypass_supports_pruning_without_fusion(monkeypatch, tmp_path):
    method = CoarseToFine64(router=_StaticRouter([]))
    monkeypatch.setattr(
        method,
        "_cached_windows",
        lambda sample, cache: ([Window(0.0, sample.duration)], "content"),
    )
    backend = _PruningTelemetryBackend()
    sample = Sample("short", "video", Path(__file__), 20.0, "event", cardinality="multi")

    result = method.run(sample, backend, tmp_path)

    assert result.telemetry["bypass"] is True
    assert result.telemetry["total_frames"] == FRAME_BUDGET
    assert len(backend.encode_calls) == len(backend.predict_calls) == 1
    assert result.telemetry["fusion_groups"] == []
    assert result.telemetry["window_telemetry"][0]["semvid_output_tokens"] is not None


def test_multi_cardinality_prompt_forces_enumerated_occurrences(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    multi = Sample("1", "v", video, 100.0, "a person drinks", cardinality="multi")
    single = Sample("1", "v", video, 100.0, "a person drinks", cardinality="single")
    multi_prompt = QwenEvidenceBackend._prompt(multi, GroundingContext(0.0, 100.0))
    single_prompt = QwenEvidenceBackend._prompt(single, GroundingContext(0.0, 100.0))
    assert "MULTIPLE" in multi_prompt and "EVERY occurrence" in multi_prompt
    assert "[[1.0, 3.0], [7.5, 9.0]]" in multi_prompt
    assert "Do NOT return the whole video" in multi_prompt
    assert "the best interval" in single_prompt and "MULTIPLE" not in single_prompt
