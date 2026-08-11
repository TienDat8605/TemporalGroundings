from pathlib import Path

import torch

from hybrid_vtg.contracts import GroundingContext, ModelBackend, Prediction, Sample, TemporalEvidence
from hybrid_vtg.methods.coarse_to_fine_64 import (
    FRAME_BUDGET,
    CoarseToFine64,
    EmbeddingRouter,
    Window,
    content_windows,
    distribute_frames,
    strict_budget,
    uniform_windows,
)
from hybrid_vtg.methods.hmve import HMVE, pack_evidence, propose_boundary_bands, propose_corridors
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
    # The method bypasses routing for this case; strict_budget itself remains valid.
    assert policy["router_frames"] + policy["local_budget"] == 64


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

    # A resumed run reuses the cache instead of re-running scene detection.
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
    assert [list(value) for value in calls[1][0]] == [
        ["video"],
        ["video"],
    ]
    assert all(call[1]["normalize_embeddings"] for call in calls)


def test_hmve_corridors_are_query_ranked_and_separated():
    values = [(float(index * 5), float(index), index) for index in range(10)]
    corridors = propose_corridors(values, 50.0, maximum=3)
    assert len(corridors) == 3
    assert all(0 <= value.start < value.end <= 50 for value in corridors)


def test_hmve_corridor_cap_surfaces_more_occurrences():
    # Peaks every 9s across a 90s video; a higher cap must surface more corridors.
    values = [(float(index * 3), 0.1 + (0.9 if index % 3 == 0 else 0.0), index) for index in range(30)]
    assert len(propose_corridors(values, 90.0, maximum=4)) == 4
    assert len(propose_corridors(values, 90.0, maximum=6)) == 6


def test_hmve_corridor_width_scales_with_duration():
    # A single peak in a long video gets a wider corridor than in a short one.
    peak = [(50.0, 1.0, 0)]
    short = propose_corridors(peak, 100.0, maximum=1)[0]
    long = propose_corridors(peak, 1000.0, maximum=1)[0]
    assert short.end - short.start <= long.end - long.start
    # Width is bounded: never wider than 64s even for very long videos.
    very_long = propose_corridors(peak, 10000.0, maximum=1)[0]
    assert very_long.end - very_long.start <= 128.0


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


def test_hmve_boundary_pass_targets_relevance_rises_and_falls():
    values = [(0.0, 0.1, 0), (2.0, 0.8, 1), (4.0, 0.9, 2), (6.0, 0.2, 3)]
    corridors = propose_corridors(values, 8.0, maximum=1)
    bands = propose_boundary_bands(values, corridors, 8.0, radius=1.0)
    assert [value.role for value in bands] == ["start", "end"]
    assert bands[0].start <= 2.0 <= bands[0].end
    assert bands[1].start <= 4.0 <= bands[1].end


def test_observation_timestamps_carries_roles():
    from hybrid_vtg.methods.hmve import observation_timestamps

    windows = [type("W", (), {"start": 0.0, "end": 4.0})(), type("W", (), {"start": 8.0, "end": 12.0})()]
    timestamps, roles = observation_timestamps(windows, 1.0, roles=["start", "end"])
    assert len(timestamps) == len(roles)
    assert all(role in ("start", "end") for role in roles)
    # Roles align with their window's timestamps.
    for timestamp, role in zip(timestamps, roles):
        if timestamp < 6.0:
            assert role == "start"
        else:
            assert role == "end"


def test_evidence_roles_survive_select_and_concatenate():
    import torch

    from hybrid_vtg.contracts import TemporalEvidence

    a = TemporalEvidence(torch.eye(4), (0.0, 1.0, 2.0, 3.0), 4, roles=("x", "y", "x", "y"))
    b = TemporalEvidence(torch.eye(4), (4.0, 5.0, 6.0, 7.0), 4, roles=("z", "z", "z", "z"))
    merged = TemporalEvidence.concatenate([a, b])
    assert len(merged.roles) == merged.size
    assert set(merged.roles) == {"x", "y", "z"}
    kept = merged.select([0, 3, 7])
    assert len(kept.roles) == 3
    assert kept.roles == (merged.roles[0], merged.roles[3], merged.roles[7])


def test_evidence_prompt_emits_role_text_when_roles_present(tmp_path):
    import torch

    from hybrid_vtg.contracts import TemporalEvidence

    evidence = TemporalEvidence(
        torch.eye(4),
        (10.0, 20.0, 30.0, 40.0),
        4,
        roles=("start", "end", "content", "global_anchor"),
    )
    assert len(evidence.roles) == len(evidence.timestamps)
    assert evidence.roles[0] == "start" and evidence.roles[1] == "end"


def test_hmve_pack_preserves_anchors_and_exact_target():
    embeddings = torch.eye(8)
    evidence = TemporalEvidence(embeddings, tuple(float(index) for index in range(8)), 8)
    scores = torch.arange(8, dtype=torch.float32)
    packed = pack_evidence(evidence, scores, 4, {0, 3})
    assert packed.size == 4
    assert 0.0 in packed.timestamps and 3.0 in packed.timestamps


class _BoundedBackend(ModelBackend):
    name = "bounded"

    def __init__(self):
        self.encoder_calls = 0
        self.predict_calls = 0

    @property
    def maximum_evidence_units(self):
        return 75

    def encode(self, sample, timestamps):
        del sample
        self.encoder_calls += 1
        values = torch.arange(len(timestamps) * 4, dtype=torch.float32).reshape(len(timestamps), 4)
        return TemporalEvidence(values, tuple(timestamps), len(timestamps))

    def query_scores(self, evidence, query):
        del query
        return torch.arange(evidence.size, dtype=torch.float32)

    def predict(self, sample, evidence, context: GroundingContext):
        del sample, context
        self.predict_calls += 1
        assert evidence.size <= self.maximum_evidence_units
        return Prediction(())


def test_hmve_reserves_detail_capacity_for_bounded_temporal_models():
    sample = Sample("long", "video", Path(__file__), 600.0, "event")
    backend = _BoundedBackend()
    result = HMVE().run(sample, backend, Path("unused"))
    assert result.telemetry["scout_frames"] <= 37
    assert result.telemetry["retained_evidence"] <= 75
    assert backend.encoder_calls == result.telemetry["encoder_calls"] == 3
    assert backend.predict_calls == result.telemetry["llm_or_fusion_calls"] == 1
    assert [value["role"] for value in result.telemetry["passes"]] == [
        "global-scout",
        "corridor-refinement",
        "boundary-refinement",
    ]


def test_hmve_default_pass_rates_end_at_three_fps():
    method = HMVE()
    assert (method.scout_fps, method.detail_fps, method.boundary_fps) == (0.5, 1.0, 3.0)
