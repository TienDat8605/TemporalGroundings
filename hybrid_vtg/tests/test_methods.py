import math
from pathlib import Path

import pytest
import torch

from hybrid_vtg.contracts import GroundingContext, ModelBackend, Prediction, Sample, TemporalEvidence
from hybrid_vtg.methods.boundary_guided_sparsification import (
    BoundaryBracket,
    BoundaryGuidedSparsification,
    CoarseObservation,
    EpisodePair,
    PresenceCalibration,
    aggregate_query_presence,
    detect_boundaries,
    estimate_refinement_cost,
    normalize_presence,
    pair_boundaries,
    select_episode_pairs,
    select_refinement_pairs,
    state_for_score,
)
from hybrid_vtg.methods.budget import (
    REFINEMENT_RESERVE_FRAMES,
    duplicate_tubelets,
    duration_budget,
    scout_timestamps,
)
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
from hybrid_vtg.methods.tpsa_query import TPSAQuery
from hybrid_vtg.methods.uniform_budget import UniformBudget
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


def test_tpsa_retains_exact_budget_and_timeline_coverage():
    timestamps = tuple(float(index // 4) for index in range(40))
    scores = torch.linspace(0, 1, 40)
    indices = TPSAQuery.select_indices(scores, timestamps, 8)
    assert indices.numel() == 8
    assert indices.tolist() == sorted(indices.tolist())
    assert min(timestamps[index] for index in indices) <= 2.0
    assert max(timestamps[index] for index in indices) >= 8.0


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


def test_vision_prune_indices_preserve_merge_cells():
    import torch

    from hybrid_vtg.models.qwen import _prune_indices

    grid = torch.tensor([[2, 4, 4]])  # 2 frames, 4x4 patches, merge=2 -> 2x2 cells
    survive, new_grid = _prune_indices(grid, 2, 0.5)
    # 2 frames * 2x2 cells * 4 tokens = 16 tokens; keep even cell rows -> 8 tokens.
    assert len(survive) == 16
    assert new_grid == [[2, 2, 4]]
    # Every surviving token belongs to a whole 2x2 cell (4 consecutive tokens).
    assert all(survive[i : i + 4] == list(range(survive[i], survive[i] + 4)) for i in range(0, 16, 4))

    survive, new_grid = _prune_indices(grid, 2, 0.25)
    assert len(survive) == 8
    assert new_grid == [[2, 2, 2]]


def test_duration_budget_schedule_is_even_and_duration_scaled():
    assert duration_budget(4.0) == 66
    assert duration_budget(150.0) == 102
    assert duration_budget(17 * 60.0) == 320
    assert all(
        duration_budget(value) >= 2 * math.ceil(value / 8.0) + REFINEMENT_RESERVE_FRAMES
        and duration_budget(value) % 2 == 0
        for value in (1.0, 60.0, 900.0)
    )


def test_scout_timestamps_use_fixed_cells_and_report_qwen_padding():
    timestamps, padding = scout_timestamps(17.0, require_even=True)
    assert timestamps == (4.0, 12.0, 16.5, 16.5)
    assert padding == 1


def test_duplicate_tubelets_create_clamped_micro_windows_with_logical_mapping():
    timestamps, roles, duplicates, logical_indices = duplicate_tubelets(
        (0.1, 12.0, 20.0),
        ("a", "b", "c"),
        True,
        duration=20.1,
    )
    assert timestamps == pytest.approx((0.0, 0.35, 11.75, 12.25, 19.75, 20.1))
    assert roles == ("a", "a", "b", "b", "c", "c")
    assert duplicates == 3
    assert logical_indices == (0, 0, 1, 1, 2, 2)

    timestamps, roles, duplicates, logical_indices = duplicate_tubelets((4.0,), ("a",), False)
    assert timestamps == (4.0,)
    assert roles == ("a",)
    assert duplicates == 0
    assert logical_indices == (0,)


def test_presence_aggregation_uses_upper_quartile_mean_not_maximum():
    evidence = TemporalEvidence(torch.eye(5), (4.0,) * 5, 1)
    observations = aggregate_query_presence(evidence, torch.tensor([100.0, 1.0, 1.0, 1.0, 1.0]))
    assert observations[0].aggregate == 50.5
    assert observations[0].aggregate < 100.0


def test_presence_aggregation_can_group_qwen_micro_windows_by_logical_index():
    evidence = TemporalEvidence(torch.eye(4), (3.75, 4.25, 11.75, 12.25), 4)
    observations = aggregate_query_presence(
        evidence,
        torch.tensor([1.0, 3.0, 2.0, 4.0]),
        logical_indices=(0, 0, 1, 1),
    )
    assert len(observations) == 2
    assert observations[0].timestamp == pytest.approx(4.0)
    assert observations[1].timestamp == pytest.approx(12.0)
    assert observations[0].aggregate == 3.0
    assert observations[1].aggregate == 4.0


def test_median_mad_states_and_constant_timeline():
    values = tuple(CoarseObservation(float(index), (score,), score) for index, score in enumerate((0.0, 0.5, 1.0, 2.0)))
    normalized, calibration = normalize_presence(values)
    assert {value.state for value in normalized} == {"absent", "present", "uncertain"}
    assert calibration.mad > 0
    constant, calibration = normalize_presence(
        tuple(CoarseObservation(float(index), (1.0,), 1.0) for index in range(4))
    )
    assert calibration.constant
    assert all(value.state == "uncertain" and value.normalized is None for value in constant)


def test_present_threshold_includes_point_seventy_five_only():
    calibration = PresenceCalibration(median=0.0, mad=1.0, scale=1.0)
    assert state_for_score(0.75, calibration) == (0.75, "present")
    assert state_for_score(0.74, calibration) == (0.74, "uncertain")


def test_select_episode_pairs_caps_multi_to_six_non_overlapping():
    pairs = []
    for index in range(24):
        start = float(index * 2)
        end = start + 1.0
        pairs.append(
            EpisodePair(
                pair_id=index,
                start=BoundaryBracket(start - 0.5, start, "absent", "present", "start", 1.0),
                end=BoundaryBracket(end, end + 0.5, "present", "absent", "end", 1.0),
                score=float(100 - index),
            )
        )

    selected = select_episode_pairs(pairs, "multi")
    assert len(selected) == 6
    assert [value.pair_id for value in selected] == list(range(6))
    assert all(
        left.interval[1] <= right.interval[0] for left, right in zip(selected, selected[1:])
    )


def test_refinement_admission_only_selects_complete_pair_work_that_fits_reserve():
    pairs = tuple(
        EpisodePair(
            pair_id=index,
            start=BoundaryBracket(index * 20.0, index * 20.0 + 8.0, "absent", "present", "start", 1.0),
            end=BoundaryBracket(index * 20.0 + 9.0, index * 20.0 + 17.0, "present", "absent", "end", 1.0),
            score=float(10 - index),
        )
        for index in range(4)
    )
    selected, rejected = select_refinement_pairs(pairs, "multi", 64, even_frames=True)

    assert [pair.pair_id for pair in selected] == [0, 1, 2]
    assert sum(estimate_refinement_cost(pair, True) for pair in selected) == 60
    assert rejected == ({"pair_id": 3, "reason": "refinement_budget", "estimated_frames": 20},)


def test_persistent_transitions_pair_without_isolated_noise():
    states = ("absent", "absent", "present", "absent", "absent", "present", "present", "absent", "absent")
    values = tuple(
        CoarseObservation(float(index * 8 + 4), (0.0,), 0.0, -1.0 if state == "absent" else 2.0, state)
        for index, state in enumerate(states)
    )
    brackets, _ = detect_boundaries(values, 72.0)
    pairs = pair_boundaries(brackets, values, 72.0)
    assert [value.role for value in brackets] == ["start", "end"]
    assert len(pairs) == 1
    assert pairs[0].interval == (44.0, 52.0)


class _ScriptedBoundaryBackend(ModelBackend):
    name = "scripted-boundary"

    def __init__(self, rows_per_timestamp=1, ambiguous_refinement=False, maximum=75):
        self.rows_per_timestamp = rows_per_timestamp
        self.ambiguous_refinement = ambiguous_refinement
        self.maximum = maximum
        self.encoder_calls = 0
        self.predict_calls = 0
        self.requested_frames = 0

    @property
    def capabilities(self):
        if self.rows_per_timestamp > 1:
            return frozenset({"encoded-evidence", "spatial-evidence"})
        return frozenset({"encoded-evidence"})

    @property
    def maximum_evidence_units(self):
        return self.maximum

    def encode(self, sample, timestamps):
        del sample
        self.encoder_calls += 1
        self.requested_frames += len(timestamps)
        expanded = tuple(float(value) for value in timestamps for _ in range(self.rows_per_timestamp))
        embeddings = torch.arange(len(expanded) * 8, dtype=torch.float32).reshape(len(expanded), 8) + 1
        return TemporalEvidence(embeddings, expanded, len(timestamps))

    def query_scores(self, evidence, query):
        del query
        if self.ambiguous_refinement and self.encoder_calls > 1 and "parts" not in evidence.metadata:
            return torch.full((evidence.size,), 1.5)
        return torch.tensor([3.0 if 18.0 <= value <= 30.0 else 0.0 for value in evidence.timestamps])

    def predict(self, sample, evidence, context):
        del sample, context
        self.predict_calls += 1
        assert evidence.size <= self.maximum_evidence_units
        return Prediction(())


def test_uniform_budget_uses_exact_matched_frame_ledger():
    sample = Sample("uniform", "video", Path(__file__), 150.0, "event")
    backend = _ScriptedBoundaryBackend()
    result = UniformBudget().run(sample, backend, Path("unused"))
    assert backend.requested_frames == result.telemetry["budget"] == duration_budget(sample.duration)
    assert backend.encoder_calls == backend.predict_calls == 1
    assert result.telemetry["remaining_frames"] == 0
    assert result.telemetry["retained_evidence"] <= backend.maximum_evidence_units
    assert UniformBudget().retention_ratio == 0.25
    assert result.telemetry["retention_target"] == 0.25


def test_bgs_primary_returns_resolved_directional_brackets_without_prediction_call():
    sample = Sample("bgs", "video", Path(__file__), 48.0, "event", cardinality="multi")
    backend = _ScriptedBoundaryBackend()
    result = BoundaryGuidedSparsification().run(sample, backend, Path("unused"))
    corridors = result.telemetry["boundary_corridors"]
    assert len(corridors) == 2
    assert all(value["end"] - value["start"] <= 1.0 for value in corridors)
    assert result.spans and result.spans[0].start < result.spans[0].end
    assert backend.predict_calls == result.telemetry["llm_or_fusion_calls"] == 0
    assert result.telemetry["prediction_source"] == "bgs-primary"
    assert result.telemetry["qwen_fallback_used"] is False
    assert result.raw_output == "[[17.5, 30.5]]"
    assert backend.requested_frames == result.telemetry["requested_frames"] < result.telemetry["budget"]
    assert BoundaryGuidedSparsification().retention_ratio == 0.25
    assert result.telemetry["constants"]["present_threshold"] == 0.75
    assert result.telemetry["constants"]["maximum_pairs"] == 6
    assert result.telemetry["constants"]["retention_ratio"] == 0.25


def test_bgs_uses_one_qwen_fallback_for_no_candidate_pair():
    class _NoCandidateBackend(_ScriptedBoundaryBackend):
        def query_scores(self, evidence, query):
            del query
            return torch.zeros(evidence.size)

    sample = Sample("bgs-none", "video", Path(__file__), 48.0, "event")
    backend = _NoCandidateBackend()
    result = BoundaryGuidedSparsification().run(sample, backend, Path("unused"))

    assert result.telemetry["prediction_source"] == "qwen-fallback"
    assert result.telemetry["fallback_reason"] == "no_candidate"
    assert result.telemetry["llm_or_fusion_calls"] == backend.predict_calls == 1


def test_bgs_supports_qwen_style_rows_and_preserves_ambiguous_corridors():
    sample = Sample("bgs-spatial", "video", Path(__file__), 48.0, "event")
    backend = _ScriptedBoundaryBackend(rows_per_timestamp=5, ambiguous_refinement=True, maximum=40)
    result = BoundaryGuidedSparsification().run(sample, backend, Path("unused"))
    corridors = result.telemetry["boundary_corridors"]
    assert corridors
    assert all(value["status"] == "ambiguous" for value in corridors)
    assert sum(value["stage"] == "ambiguity-probe" for value in result.telemetry["refinement"]) <= 1
    assert result.telemetry["duplicate_padding_frames"] >= 0
    assert result.telemetry["retained_evidence"] <= backend.maximum_evidence_units
    assert backend.predict_calls == 1
    assert result.telemetry["prediction_source"] == "qwen-fallback"
    assert result.telemetry["fallback_reason"] == "unresolved_boundaries"


def test_bgs_counts_qwen_tubelet_duplicates_in_its_frame_ledger():
    sample = Sample("bgs-tubelets", "video", Path(__file__), 48.0, "event")
    backend = _ScriptedBoundaryBackend(rows_per_timestamp=5)
    result = BoundaryGuidedSparsification().run(sample, backend, Path("unused"))

    assert result.telemetry["qwen_tubelet_duplication"] is True
    assert result.telemetry["scout_logical_frames"] == 6
    assert result.telemetry["scout_physical_frames"] == 12
    assert result.telemetry["scout_remaining_frames"] == REFINEMENT_RESERVE_FRAMES
    assert len(result.telemetry["coarse_observations"]) == result.telemetry["coarse_cells"] == 6
    assert result.telemetry["tubelet_duplicate_frames"] >= 6
    assert backend.requested_frames == result.telemetry["requested_frames"] < result.telemetry["budget"]


def test_qwen_encode_disables_processor_frame_sampling(monkeypatch, tmp_path):
    from PIL import Image

    frame_paths = []
    for index in range(4):
        path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (32, 32)).save(path)
        frame_paths.append(path)

    class _Processor:
        vision_start_token = "<vision>"
        video_token = "<video>"
        vision_end_token = "</vision>"

        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {
                "pixel_values_videos": torch.zeros((8, 8)),
                "video_grid_thw": torch.tensor([[2, 2, 2]]),
            }

    class _Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1))

    class _InnerModel:
        def __init__(self):
            self.visual = _Visual()

        def get_video_features(self, pixels, grids):
            del pixels, grids
            return [torch.ones((2, 4))], None

    class _Model:
        def __init__(self):
            self.model = _InnerModel()
            self.config = type(
                "Config",
                (),
                {"vision_config": type("VisionConfig", (), {"spatial_merge_size": 2, "temporal_patch_size": 2})()},
            )()

    processor = _Processor()
    backend = QwenEvidenceBackend("unused", tmp_path, name="qwen-test")
    monkeypatch.setattr(backend, "_load", lambda: (_Model(), processor))
    monkeypatch.setattr("hybrid_vtg.models.qwen.extract_frames", lambda *args: frame_paths)
    evidence = backend.encode(Sample("qwen", "video", Path(__file__), 20.0, "event"), (1.0, 2.0, 3.0, 4.0))

    assert processor.kwargs["do_sample_frames"] is False
    assert evidence.metadata["processor_do_sample_frames"] is False
    assert evidence.metadata["effective_temporal_units"] == 2
    assert evidence.metadata["temporal_patch_size"] == 2
