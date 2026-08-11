"""Strict 64-frame embedding-window coarse-to-fine search."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample
from ...media import extract_frames, uniform_timestamps
from ...postprocess import consolidate_spans

FRAME_BUDGET = 64


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


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

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Run on CPU: the 4B grounder stays resident on the GPU for the whole run, so
            # the 2B router must not compete with it for GPU memory (otherwise OOM).
            self._model = SentenceTransformer(self.model_id, device="cpu")
            if not self._model.supports("video"):
                raise RuntimeError(f"embedding model does not support video input: {self.model_id}")
        return self._model

    def rank(
        self,
        sample: Sample,
        windows: Sequence[Window],
        frames_per_window: int,
        cache_dir: Path,
    ) -> list[float]:
        model = self._load()
        query = model.encode(
            [sample.query],
            prompt="Represent this text for retrieving matching temporal video windows. ",
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        videos = []
        for index, window in enumerate(windows):
            timestamps = uniform_timestamps(window.start, window.end, frames_per_window)
            frames = extract_frames(sample.video_path, timestamps, cache_dir / f"router-{index}")
            videos.append({"video": [str(path) for path in frames]})
        embeddings = model.encode(
            videos,
            prompt="Represent this video window for retrieval by a textual event description. ",
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(value) for value in (embeddings.float() @ query.float()).tolist()]


class CoarseToFine64(Method):
    name = "coarse-to-fine-64"

    def __init__(self, router: EmbeddingRouter | None = None) -> None:
        self.router = router or EmbeddingRouter()
        self._prepare_root: Path | None = None

    def prepare(self, samples: Sequence[Sample], cache_root: Path) -> None:
        """Run PySceneDetect once for every pending sample, before the GPU model loads.

        Scene detection is CPU-only and independent of the query, so it is done in one
        batch pass up front. Results are cached on disk so a resumed run reuses them.
        """
        self._prepare_root = cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            cache_path = cache_root / f"{sample.id}.json"
            if cache_path.is_file():
                continue
            windows, source = content_windows(sample.video_path, sample.duration)
            cache_path.write_text(
                json.dumps(
                    {
                        "source": source,
                        "windows": [{"start": value.start, "end": value.end} for value in windows],
                    }
                ),
                encoding="utf-8",
            )

    def _cached_windows(self, sample: Sample, cache_dir: Path) -> tuple[list[Window], str]:
        cache_path = self._prepare_root / f"{sample.id}.json" if self._prepare_root else None
        if cache_path is None or not cache_path.is_file():
            cache_path = cache_dir / f"{sample.id}.json"
        if cache_path.is_file():
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            windows = [Window(float(item["start"]), float(item["end"])) for item in value["windows"]]
            return windows, str(value["source"])
        windows, source = content_windows(sample.video_path, sample.duration)
        return windows, source

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        self.validate_model(model)
        windows, source = self._cached_windows(sample, cache_dir)
        if len(windows) == 1:
            timestamps = uniform_timestamps(0.0, sample.duration, FRAME_BUDGET)
            evidence = model.encode(sample, timestamps)
            result = model.predict(sample, evidence, GroundingContext(0.0, sample.duration))
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
                },
            )

        routed, policy = strict_budget(windows)
        scores = self.router.rank(
            sample,
            routed,
            policy["router_frames_per_window"],
            cache_dir / "router",
        )
        selected = sorted(range(len(routed)), key=lambda index: (-scores[index], index))[: policy["selected_windows"]]
        allocations = distribute_frames(policy["local_budget"], len(selected))
        predictions: list[tuple[int, Prediction]] = []
        for index, count in zip(selected, allocations):
            window = routed[index]
            evidence = model.encode(sample, uniform_timestamps(window.start, window.end, count))
            predictions.append(
                (
                    index,
                    model.predict(
                        sample,
                        evidence,
                        GroundingContext(window.start, window.end),
                    ),
                )
            )

        candidates = [span for _, prediction in predictions for span in prediction.spans]
        if sample.cardinality == "multi":
            spans = consolidate_spans(candidates, sample.duration)
        else:
            spans = ()
            for index in selected:
                prediction = next(value for key, value in predictions if key == index)
                if prediction.spans:
                    spans = (prediction.spans[0],)
                    break
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
                "grounder_frames": sum(allocations),
                "total_frames": total,
            },
        )
