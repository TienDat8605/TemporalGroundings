"""Orchestration for the hierarchical training-free VTG pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .coarse_encoder import FrozenSiglipEncoder
from .config import PipelineConfig
from .index import build_index, cache_path, load_index, save_index, video_fingerprint
from .refinement import refine_prediction
from .semvid_bridge import SemVIDGrounder
from .temporal import interval_evidence_score, route
from .types import Component, Sample, TemporalRoute


class HybridVTGPipeline:
    def __init__(self, config: PipelineConfig, cache_dir: Path, semvid_root: Path | None = None) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.coarse_encoder: FrozenSiglipEncoder | None = None
        self.grounder = SemVIDGrounder(config.semvid, semvid_root)

    def _encoder(self) -> FrozenSiglipEncoder:
        if self.coarse_encoder is None:
            self.coarse_encoder = FrozenSiglipEncoder(
                self.config.coarse.checkpoint, batch_size=self.config.coarse.batch_size,
            )
        return self.coarse_encoder

    def _index(self, sample: Sample):
        path = cache_path(self.cache_dir, sample.video_path, self.config.coarse)
        if path.is_file():
            index = load_index(path)
            if (
                index.fingerprint == video_fingerprint(sample.video_path)
                and abs(index.duration - sample.duration) < 0.5
            ):
                return index, True
        index = build_index(sample.video_path, sample.duration, self.config.coarse, self._encoder())
        save_index(path, index)
        return index, False

    def run_sample(self, sample: Sample) -> dict[str, Any]:
        sample.validate()
        started = perf_counter()
        self._reset_peak_memory()
        index = None
        query_embedding = None
        candidates = []
        cache_hit = False
        index_seconds = route_seconds = 0.0
        if self.config.coarse.enabled:
            stage_started = perf_counter()
            index, cache_hit = self._index(sample)
            index_seconds = perf_counter() - stage_started
            stage_started = perf_counter()
            query_embedding = self._encoder().encode_text(sample.query)
            temporal_route, candidates = route(index, query_embedding, self.config.coarse)
            route_seconds = perf_counter() - stage_started
        else:
            whole_video = Component(0.0, sample.duration, 1.0)
            temporal_route = TemporalRoute(
                components=(whole_video,), selected_candidates=(), confidence_margin=0.0,
                low_confidence_fallback=False, retained_union_seconds=sample.duration,
            )
        ordered_components = sorted(temporal_route.components, key=lambda value: (-value.score, value.start))
        stage_started = perf_counter()
        proposals = []
        proposal_errors = []
        for component in ordered_components:
            try:
                component_prediction = self.grounder.ground(sample, component)
            except ValueError as error:
                proposal_errors.append({"component": asdict(component), "error": str(error)})
                continue
            selection_score = component.score
            if index is not None and query_embedding is not None:
                selection_score = interval_evidence_score(
                    index, query_embedding, component_prediction.interval, self.config.coarse.mean_weight,
                )
                if selection_score <= -1.0:
                    selection_score = component.score
            proposals.append((selection_score, component_prediction))
        if not proposals:
            raise RuntimeError(f"all routed component predictions failed: {proposal_errors}")
        proposals.sort(key=lambda value: (-value[0], value[1].interval[0]))
        prediction = proposals[0][1]
        ground_seconds = perf_counter() - stage_started
        refinement = None
        refinement_seconds = 0.0
        final_interval = prediction.interval
        if self.config.refinement.enabled:
            if query_embedding is None:
                query_embedding = self._encoder().encode_text(sample.query)
            stage_started = perf_counter()
            refinement = refine_prediction(
                sample, prediction, self._encoder(), query_embedding, self.config.refinement,
            )
            refinement_seconds = perf_counter() - stage_started
            final_interval = refinement.interval
        prediction_value = prediction.to_dict()
        prediction_value["coarse_interval"] = list(prediction.interval)
        prediction_value["interval"] = list(final_interval)
        return {
            "id": sample.id,
            "video": sample.video,
            "query": sample.query,
            "duration": sample.duration,
            "targets": [list(value) for value in sample.targets],
            "prediction": prediction_value,
            "component_predictions": [
                {**value.to_dict(), "selection_score": score} for score, value in proposals
            ],
            "component_errors": proposal_errors,
            "route": {
                **asdict(temporal_route),
                "retained_fraction": temporal_route.retained_union_seconds / sample.duration,
                "candidates": [asdict(candidates[index]) for index in temporal_route.selected_candidates],
            },
            "refinement": asdict(refinement) if refinement else None,
            "cache_hit": cache_hit,
            "efficiency": {
                "coarse_frames": len(index.timestamps) if index is not None else 0,
                "effective_coarse_fps": index.fps if index is not None else 0.0,
                "expert_seconds": sum(component.duration for component in ordered_components),
                "semvid_original_tokens": sum(
                    int(value.semvid_stats.get("orig_video_tokens", 0)) for _, value in proposals
                ),
                "semvid_retained_tokens": sum(
                    int(value.semvid_stats.get("kept_video_tokens", 0)) for _, value in proposals
                ),
                "timing_seconds": {
                    "index": index_seconds,
                    "route": route_seconds,
                    "ground": ground_seconds,
                    "refine": refinement_seconds,
                    "total": perf_counter() - started,
                },
                "peak_gpu_memory_bytes": self._peak_memory(),
            },
        }

    @staticmethod
    def _reset_peak_memory() -> None:
        try:
            import torch
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _peak_memory() -> dict[str, int]:
        try:
            import torch
            return {
                str(device): int(torch.cuda.max_memory_allocated(device))
                for device in range(torch.cuda.device_count())
            }
        except (ImportError, RuntimeError):
            return {}
