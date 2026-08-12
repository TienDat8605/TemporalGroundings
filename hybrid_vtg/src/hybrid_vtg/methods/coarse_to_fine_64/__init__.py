"""Strict 64-frame embedding-window coarse-to-fine search."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample, ScoredSpan
from ...media import extract_frames, uniform_timestamps
from ...postprocess import temporal_iou

FRAME_BUDGET = 64
SCENE_CACHE_SCHEMA = 1
SCENE_POLICY = "content-t27-s4-window20-60-v1"
ROUTER_CACHE_SCHEMA = 1
ROUTER_POLICY = "qwen3-vl-embedding-normalized-v1"
ROUTER_TEXT_PROMPT = "Represent this text for retrieving matching temporal video windows. "
ROUTER_VIDEO_PROMPT = "Represent this video window for retrieval by a textual event description. "


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class WindowCandidate:
    """One absolute-time prediction with its local-window provenance."""

    window_index: int
    router_score: float
    span: ScoredSpan


def _span_dict(span: ScoredSpan) -> dict[str, float]:
    return {"start": span.start, "end": span.end, "score": span.score}


def fuse_cross_window_spans(
    candidates: Sequence[WindowCandidate],
    duration: float,
    *,
    duplicate_iou: float = 0.6,
) -> tuple[tuple[ScoredSpan, ...], list[dict[str, Any]]]:
    """Fuse duplicate spans across windows while preserving local occurrences.

    Router score determines seed order and the softmax weight used for boundaries.
    A group can contain no more than one candidate from any source window. Candidates
    from the same window are therefore never treated as duplicates of one another.
    """
    if not 0.0 <= duplicate_iou <= 1.0:
        raise ValueError("duplicate_iou must be between 0 and 1")

    clipped: list[tuple[int, WindowCandidate]] = []
    for order, candidate in enumerate(candidates):
        span = candidate.span.clipped(duration)
        if span is not None:
            clipped.append(
                (
                    order,
                    WindowCandidate(candidate.window_index, candidate.router_score, span),
                )
            )
    ranked = sorted(
        clipped,
        key=lambda item: (
            -item[1].router_score,
            item[1].window_index,
            item[1].span.start,
            item[1].span.end,
            item[0],
        ),
    )
    remaining = {order for order, _ in ranked}
    fused: list[ScoredSpan] = []
    groups: list[dict[str, Any]] = []

    for seed_order, seed in ranked:
        if seed_order not in remaining:
            continue
        members = [(seed_order, seed)]
        remaining.remove(seed_order)
        other_windows = sorted(
            {candidate.window_index for order, candidate in ranked if order in remaining}
            - {seed.window_index}
        )
        for window_index in other_windows:
            matches = [
                (order, candidate, temporal_iou(seed.span, candidate.span))
                for order, candidate in ranked
                if order in remaining and candidate.window_index == window_index
            ]
            matches = [value for value in matches if value[2] > duplicate_iou]
            if not matches:
                continue
            order, candidate, _ = max(
                matches,
                key=lambda value: (
                    value[2],
                    value[1].router_score,
                    value[1].span.score,
                    -value[0],
                ),
            )
            members.append((order, candidate))
            remaining.remove(order)

        maximum_score = max(candidate.router_score for _, candidate in members)
        weights = [math.exp(candidate.router_score - maximum_score) for _, candidate in members]
        denominator = sum(weights)
        normalized = [weight / denominator for weight in weights]
        span = ScoredSpan(
            sum(weight * candidate.span.start for weight, (_, candidate) in zip(normalized, members)),
            sum(weight * candidate.span.end for weight, (_, candidate) in zip(normalized, members)),
            sum(weight * candidate.span.score for weight, (_, candidate) in zip(normalized, members)),
        )
        fused.append(span)
        groups.append(
            {
                "group_id": len(groups),
                "fused": len(members) > 1,
                "output_span": _span_dict(span),
                "members": [
                    {
                        "candidate_id": order,
                        "window_index": candidate.window_index,
                        "router_score": candidate.router_score,
                        "softmax_weight": weight,
                        "span": _span_dict(candidate.span),
                    }
                    for weight, (order, candidate) in zip(normalized, members)
                ],
            }
        )

    chronology = sorted(
        zip(fused, groups),
        key=lambda value: (value[0].start, value[0].end, -value[0].score),
    )
    chronological_spans = tuple(value[0] for value in chronology)
    chronological_groups = []
    for group_id, (_, group) in enumerate(chronology):
        chronological_groups.append({**group, "group_id": group_id})
    return chronological_spans, chronological_groups


def uniform_windows(duration: float, length: float = 45.0, overlap: float = 4.0) -> list[Window]:
    if duration <= length:
        return [Window(0.0, duration)]
    windows, start = [], 0.0
    hop = max(1.0, length - overlap)
    while start < duration:
        end = min(duration, start + length)
        if 0 < duration - end < 20.0:
            end = duration
        windows.append(Window(start, end))
        if end >= duration:
            break
        start += hop
    return windows


def windows_from_boundaries(
    duration: float,
    boundaries: Sequence[float],
    minimum: float = 20.0,
    maximum: float = 60.0,
) -> list[Window]:
    points = sorted({0.0, duration, *(min(duration, max(0.0, value)) for value in boundaries)})
    windows, cursor = [], 0.0
    while cursor < duration - 1e-6:
        valid = [point for point in points if cursor + minimum <= point <= cursor + maximum]
        end = max(valid) if valid else min(duration, cursor + 45.0)
        if 0 < duration - end < minimum:
            end = duration
        start = cursor if not windows else max(0.0, cursor - 2.0)
        if end - start > maximum:
            start = end - maximum
        windows.append(Window(start, end))
        if end >= duration:
            break
        cursor = end
    return windows


def content_windows(video_path: Path, duration: float) -> tuple[list[Window], str]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path), backend="opencv")
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video, show_progress=False, frame_skip=4)
        boundaries = [scene[1].get_seconds() for scene in manager.get_scene_list(start_in_scene=True)[:-1]]
        if not boundaries:
            raise RuntimeError("no scene boundaries")
        return windows_from_boundaries(duration, boundaries), "content"
    except Exception:
        return uniform_windows(duration), "uniform-fallback"


def scene_cache_path(sample: Sample, cache_root: Path) -> Path:
    """Return a query- and run-independent cache path for one video revision."""
    path = sample.video_path.resolve()
    stat = path.stat()
    identity = {
        "schema": SCENE_CACHE_SCHEMA,
        "policy": SCENE_POLICY,
        "video_path": str(path),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "duration": round(sample.duration, 6),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_root / "scenes" / f"{digest}.json"


def cached_content_windows(sample: Sample, cache_root: Path) -> tuple[list[Window], str]:
    cache_path = scene_cache_path(sample, cache_root)
    if cache_path.is_file():
        try:
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            if value.get("schema") != SCENE_CACHE_SCHEMA or value.get("policy") != SCENE_POLICY:
                raise ValueError("stale scene-window cache policy")
            windows = [Window(float(item["start"]), float(item["end"])) for item in value["windows"]]
            if not windows or any(window.start < 0 or window.end <= window.start for window in windows):
                raise ValueError("invalid cached scene windows")
            return windows, str(value["source"])
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            pass

    windows, source = content_windows(sample.video_path, sample.duration)
    payload = {
        "schema": SCENE_CACHE_SCHEMA,
        "policy": SCENE_POLICY,
        "video_path": str(sample.video_path.resolve()),
        "duration": sample.duration,
        "source": source,
        "windows": [{"start": value.start, "end": value.end} for value in windows],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(cache_path)
    return windows, source


def retained_window_count(count: int) -> int:
    return min(count, min(8, max(2, math.ceil(math.sqrt(count))))) if count else 0


def coalesce_windows(windows: Sequence[Window], maximum: int) -> list[Window]:
    if len(windows) <= maximum:
        return list(windows)
    groups = np.array_split(np.arange(len(windows)), maximum)
    return [Window(windows[int(group[0])].start, windows[int(group[-1])].end) for group in groups]


def strict_budget(windows: Sequence[Window]) -> tuple[list[Window], dict[str, int]]:
    feasible = len(windows)
    while feasible > 1:
        selected = retained_window_count(feasible)
        minimum_router_frames = feasible * (2 if feasible % 2 else 1)
        if minimum_router_frames + selected * 2 <= FRAME_BUDGET:
            break
        feasible -= 1
    routed = coalesce_windows(windows, feasible)
    selected = retained_window_count(len(routed))
    router_per_window = max(
        value
        for value in range(1, 5)
        if value * len(routed) + selected * 2 <= FRAME_BUDGET and value * len(routed) % 2 == 0
    )
    router_frames = router_per_window * len(routed)
    local_budget = FRAME_BUDGET - router_frames
    return routed, {
        "selected_windows": selected,
        "router_frames_per_window": router_per_window,
        "router_frames": router_frames,
        "local_budget": local_budget,
        "unused_frames": 0,
    }


def distribute_frames(total: int, count: int) -> list[int]:
    units, remainder = divmod(total // 2, count)
    return [(units + int(index < remainder)) * 2 for index in range(count)]


class EmbeddingRouter:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-Embedding-2B") -> None:
        self.model_id = model_id
        self._model: Any = None
        self.last_telemetry: dict[str, Any] = {}

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Run on CPU: the 4B grounder stays resident on the GPU for the whole run, so
            # the 2B router must not compete with it for GPU memory (otherwise OOM).
            self._model = SentenceTransformer(self.model_id, device="cpu")
            if not self._model.supports("video"):
                raise RuntimeError(f"embedding model does not support video input: {self.model_id}")
        return self._model

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().float().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _read_cache(path: Path, *, rows: int | None = None) -> np.ndarray | None:
        try:
            with np.load(path, allow_pickle=False) as cached:
                value = np.asarray(cached["embeddings"], dtype=np.float32)
            if value.ndim != 2 or value.shape[1] == 0 or (rows is not None and value.shape[0] != rows):
                raise ValueError("invalid embedding shape")
            if not np.isfinite(value).all():
                raise ValueError("non-finite cached embedding")
            return value
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return None

    @staticmethod
    def _write_cache(path: Path, embeddings: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, embeddings=embeddings)
        temporary.replace(path)

    def _query_cache_path(self, query: str, cache_root: Path) -> Path:
        identity = {
            "schema": ROUTER_CACHE_SCHEMA,
            "policy": ROUTER_POLICY,
            "model_id": self.model_id,
            "prompt": ROUTER_TEXT_PROMPT,
            "query": query,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        return cache_root / "router" / "queries" / f"{digest}.npz"

    def _video_cache_path(
        self,
        sample: Sample,
        windows: Sequence[Window],
        frames_per_window: int,
        cache_root: Path,
    ) -> tuple[Path, list[tuple[float, ...]]]:
        path = sample.video_path.resolve()
        stat = path.stat()
        timestamps = [
            uniform_timestamps(window.start, window.end, frames_per_window) for window in windows
        ]
        identity = {
            "schema": ROUTER_CACHE_SCHEMA,
            "policy": ROUTER_POLICY,
            "model_id": self.model_id,
            "prompt": ROUTER_VIDEO_PROMPT,
            "video_path": str(path),
            "video_size": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "duration": round(sample.duration, 6),
            "frames_per_window": frames_per_window,
            "windows": [
                {
                    "start": round(window.start, 6),
                    "end": round(window.end, 6),
                    "timestamps": [round(value, 6) for value in values],
                }
                for window, values in zip(windows, timestamps)
            ],
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        return cache_root / "router" / "videos" / f"{digest}.npz", timestamps

    def rank(
        self,
        sample: Sample,
        windows: Sequence[Window],
        frames_per_window: int,
        cache_root: Path,
    ) -> list[float]:
        query_path = self._query_cache_path(sample.query, cache_root)
        video_path, timestamps = self._video_cache_path(
            sample,
            windows,
            frames_per_window,
            cache_root,
        )
        query_matrix = self._read_cache(query_path, rows=1)
        video_embeddings = self._read_cache(video_path, rows=len(windows))
        query_hit = query_matrix is not None
        video_hit = video_embeddings is not None
        if (
            query_matrix is not None
            and video_embeddings is not None
            and query_matrix.shape[1] != video_embeddings.shape[1]
        ):
            query_matrix = None
            video_embeddings = None
            query_hit = video_hit = False

        model = None
        if query_matrix is None:
            model = self._load()
            query_matrix = self._array(
                model.encode(
                    [sample.query],
                    prompt=ROUTER_TEXT_PROMPT,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            if query_matrix.ndim != 2 or query_matrix.shape[0] != 1 or query_matrix.shape[1] == 0:
                raise RuntimeError(f"router returned invalid query embeddings: {query_matrix.shape}")
            self._write_cache(query_path, query_matrix)
        if video_embeddings is None:
            if model is None:
                model = self._load()
            videos = []
            frame_root = cache_root / "router" / "frames" / video_path.stem
            for index, values in enumerate(timestamps):
                frames = extract_frames(sample.video_path, values, frame_root / f"window-{index}")
                videos.append({"video": [str(path) for path in frames]})
            video_embeddings = self._array(
                model.encode(
                    videos,
                    prompt=ROUTER_VIDEO_PROMPT,
                    batch_size=1,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            if (
                video_embeddings.ndim != 2
                or video_embeddings.shape[0] != len(windows)
                or video_embeddings.shape[1] == 0
            ):
                raise RuntimeError(f"router returned invalid video embeddings: {video_embeddings.shape}")
            self._write_cache(video_path, video_embeddings)
        if query_matrix.shape != (1, video_embeddings.shape[1]):
            raise RuntimeError(
                f"router embedding dimensions differ: query {query_matrix.shape}, video {video_embeddings.shape}"
            )
        self.last_telemetry = {
            "query_embedding_cache_hit": query_hit,
            "video_embedding_cache_hit": video_hit,
            "query_embedding_cache": str(query_path),
            "video_embedding_cache": str(video_path),
            "embedding_model": self.model_id,
            "embedding_policy": ROUTER_POLICY,
        }
        return [float(value) for value in (video_embeddings @ query_matrix[0]).tolist()]


class CoarseToFine64(Method):
    name = "coarse-to-fine-64"

    def __init__(self, router: EmbeddingRouter | None = None) -> None:
        self.router = router or EmbeddingRouter()
        self._prepare_root: Path | None = None

    def prepare(self, samples: Sequence[Sample], cache_root: Path) -> None:
        """Run PySceneDetect once per unique video revision before model loading.

        Scene detection is CPU-only and independent of the query, so it is done in one
        batch pass up front. The shared, video-keyed cache is reused across queries,
        seeds, model/pruning variants, reruns, and separate method instances.
        """
        self._prepare_root = cache_root
        prepared: set[Path] = set()
        for sample in samples:
            cache_path = scene_cache_path(sample, cache_root)
            if cache_path in prepared:
                continue
            cached_content_windows(sample, cache_root)
            prepared.add(cache_path)

    def _cached_windows(self, sample: Sample, cache_dir: Path) -> tuple[list[Window], str]:
        return cached_content_windows(sample, self._prepare_root or cache_dir)

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        self.validate_model(model)
        windows, source = self._cached_windows(sample, cache_dir)
        if len(windows) == 1:
            timestamps = uniform_timestamps(0.0, sample.duration, FRAME_BUDGET)
            encode_started = perf_counter()
            evidence = model.encode(sample, timestamps)
            encode_seconds = perf_counter() - encode_started
            predict_started = perf_counter()
            result = model.predict(sample, evidence, GroundingContext(0.0, sample.duration))
            predict_seconds = perf_counter() - predict_started
            semvid = result.telemetry.get("semvid", {})
            return Prediction(
                result.spans,
                result.raw_output,
                {
                    **result.telemetry,
                    "window_source": source,
                    "bypass": True,
                    "router_frames": 0,
                    "grounder_frames": FRAME_BUDGET,
                    "total_frames": FRAME_BUDGET,
                    "window_telemetry": [
                        {
                            "window_index": 0,
                            "window": {"start": 0.0, "end": sample.duration},
                            "router_score": None,
                            "selected": True,
                            "allocated_frames": FRAME_BUDGET,
                            "encoder_dense_tokens": evidence.metadata.get(
                                "dense_evidence_units", evidence.size
                            ),
                            "encoder_retained_tokens": evidence.metadata.get(
                                "encoder_retained_evidence_units", evidence.size
                            ),
                            "semvid_input_tokens": semvid.get("semvid_input_evidence_units"),
                            "semvid_output_tokens": semvid.get("semvid_retained_evidence_units"),
                            "llm_input_tokens": result.telemetry.get("llm_input_tokens"),
                            "llm_output_tokens": result.telemetry.get("llm_output_tokens"),
                            "timing": {
                                "encode_seconds": encode_seconds,
                                "predict_seconds": predict_seconds,
                                "total_seconds": encode_seconds + predict_seconds,
                            },
                            "local_spans": [
                                _span_dict(
                                    ScoredSpan(span.start, span.end, span.score)
                                )
                                for span in result.spans
                            ],
                            "absolute_spans": [_span_dict(span) for span in result.spans],
                            "fusion_group_ids": [],
                        }
                    ],
                    "fusion_groups": [],
                },
            )

        routed, policy = strict_budget(windows)
        router_started = perf_counter()
        scores = self.router.rank(
            sample,
            routed,
            policy["router_frames_per_window"],
            self._prepare_root or cache_dir,
        )
        router_seconds = perf_counter() - router_started
        selected = sorted(range(len(routed)), key=lambda index: (-scores[index], index))[: policy["selected_windows"]]
        allocations = distribute_frames(policy["local_budget"], len(selected))
        predictions: list[tuple[int, Prediction]] = []
        window_telemetry: list[dict[str, Any]] = [
            {
                "window_index": index,
                "window": {"start": window.start, "end": window.end},
                "router_score": scores[index],
                "selected": False,
                "allocated_frames": 0,
                "encoder_dense_tokens": None,
                "encoder_retained_tokens": None,
                "semvid_input_tokens": None,
                "semvid_output_tokens": None,
                "llm_input_tokens": None,
                "llm_output_tokens": None,
                "timing": None,
                "local_spans": [],
                "absolute_spans": [],
                "fusion_group_ids": [],
            }
            for index, window in enumerate(routed)
        ]
        for index, count in zip(selected, allocations):
            window = routed[index]
            encode_started = perf_counter()
            evidence = model.encode(sample, uniform_timestamps(window.start, window.end, count))
            encode_seconds = perf_counter() - encode_started
            predict_started = perf_counter()
            prediction = model.predict(
                sample,
                evidence,
                GroundingContext(window.start, window.end),
            )
            predict_seconds = perf_counter() - predict_started
            predictions.append((index, prediction))
            semvid = prediction.telemetry.get("semvid", {})
            window_telemetry[index].update(
                {
                    "selected": True,
                    "allocated_frames": count,
                    "encoder_dense_tokens": evidence.metadata.get(
                        "dense_evidence_units", evidence.size
                    ),
                    "encoder_retained_tokens": evidence.metadata.get(
                        "encoder_retained_evidence_units", evidence.size
                    ),
                    "semvid_input_tokens": semvid.get("semvid_input_evidence_units"),
                    "semvid_output_tokens": semvid.get("semvid_retained_evidence_units"),
                    "llm_input_tokens": prediction.telemetry.get("llm_input_tokens"),
                    "llm_output_tokens": prediction.telemetry.get("llm_output_tokens"),
                    "timing": {
                        "encode_seconds": encode_seconds,
                        "predict_seconds": predict_seconds,
                        "total_seconds": encode_seconds + predict_seconds,
                    },
                    "local_spans": [
                        _span_dict(
                            ScoredSpan(
                                span.start - window.start,
                                span.end - window.start,
                                span.score,
                            )
                        )
                        for span in prediction.spans
                    ],
                    "absolute_spans": [_span_dict(span) for span in prediction.spans],
                    "raw_output": prediction.raw_output,
                }
            )

        candidates = [
            WindowCandidate(index, scores[index], span)
            for index, prediction in predictions
            for span in prediction.spans
        ]
        if sample.cardinality == "multi":
            spans, fusion_groups = fuse_cross_window_spans(candidates, sample.duration)
        else:
            spans = ()
            fusion_groups = []
            for index in selected:
                prediction = next(value for key, value in predictions if key == index)
                if prediction.spans:
                    clipped = prediction.spans[0].clipped(sample.duration)
                    spans = (clipped,) if clipped is not None else ()
                    if clipped is not None:
                        fusion_groups = [
                            {
                                "group_id": 0,
                                "fused": False,
                                "output_span": _span_dict(clipped),
                                "members": [
                                    {
                                        "candidate_id": 0,
                                        "window_index": index,
                                        "router_score": scores[index],
                                        "softmax_weight": 1.0,
                                        "span": _span_dict(clipped),
                                    }
                                ],
                            }
                        ]
                    break
        for group in fusion_groups:
            for member in group["members"]:
                window_telemetry[member["window_index"]]["fusion_group_ids"].append(
                    group["group_id"]
                )
        total = policy["router_frames"] + sum(allocations)
        if total > FRAME_BUDGET:
            raise AssertionError(f"coarse-to-fine exceeded {FRAME_BUDGET} frames: {total}")
        return Prediction(
            spans,
            "",
            {
                "window_source": source,
                "bypass": False,
                "windows": [{"start": value.start, "end": value.end} for value in routed],
                "scores": scores,
                "selected": selected,
                "allocations": allocations,
                **policy,
                "router_seconds": router_seconds,
                "router_cache": dict(getattr(self.router, "last_telemetry", {})),
                "grounder_frames": sum(allocations),
                "total_frames": total,
                "window_telemetry": window_telemetry,
                "fusion_iou_threshold": 0.6,
                "fusion_groups": fusion_groups,
            },
        )
