"""Orchestration for the hierarchical training-free VTG pipeline."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Sequence

from .coarse_encoder import FrozenSiglipEncoder
from .config import PipelineConfig
from .index import build_index, cache_path, load_index, save_index, video_fingerprint
from .refinement import decide_refinement, refine_prediction
from .semvid_bridge import (
    EventAbsentError, GroundingOutputError, GroundingRequest, PreparedGroundingBatch,
    SemVIDGrounder,
)
from .temporal import interval_boundary_quality, route
from .types import Component, GroundingPrediction, Sample, TemporalRoute


@dataclass
class _SampleContext:
    sample: Sample
    started: float
    index: Any = None
    query_embedding: Any = None
    candidates: list[Any] = field(default_factory=list)
    cache_hit: bool = False
    index_seconds: float = 0.0
    route_seconds: float = 0.0
    temporal_route: TemporalRoute | None = None
    ordered_components: tuple[Component, ...] = ()
    proposals: list[tuple[float, GroundingPrediction, dict[str, float], float]] = field(default_factory=list)
    proposal_errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _ComponentTask:
    context: _SampleContext
    request: GroundingRequest


def _proposal_sort_key(
    value: tuple[float, GroundingPrediction, dict[str, float], float],
) -> tuple[float, float, float, float]:
    """Presence dominates boundary quality and the coarse retrieval prior."""
    quality_score, prediction, _, _ = value
    return (
        -prediction.presence_score,
        -quality_score,
        -prediction.component.score,
        prediction.interval[0],
    )


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

    def _route_sample(self, sample: Sample) -> _SampleContext:
        sample.validate()
        context = _SampleContext(sample=sample, started=perf_counter())
        if self.config.coarse.enabled:
            stage_started = perf_counter()
            context.index, context.cache_hit = self._index(sample)
            context.index_seconds = perf_counter() - stage_started
            stage_started = perf_counter()
            context.query_embedding = self._encoder().encode_text(sample.query)
            context.temporal_route, context.candidates = route(
                context.index, context.query_embedding, self.config.coarse,
            )
            context.route_seconds = perf_counter() - stage_started
        else:
            whole_video = Component(0.0, sample.duration, 1.0)
            context.temporal_route = TemporalRoute(
                components=(whole_video,), selected_candidates=(), confidence_margin=0.0,
                low_confidence_fallback=False, retained_union_seconds=sample.duration,
            )
        context.ordered_components = tuple(sorted(
            context.temporal_route.components, key=lambda value: (-value.score, value.start),
        ))
        return context

    def _component_batches(self, contexts: Sequence[_SampleContext]) -> list[list[_ComponentTask]]:
        tasks = [
            _ComponentTask(context, GroundingRequest(context.sample, component))
            for context in contexts for component in context.ordered_components
        ]
        # Similar duration is a stable proxy for visual-token load before decoding.
        tasks.sort(key=lambda task: (task.request.estimated_visual_load, task.context.sample.id))
        size = self.config.semvid.batch_size
        return [tasks[start:start + size] for start in range(0, len(tasks), size)]

    def _prepare_tasks(self, tasks: Sequence[_ComponentTask]) -> PreparedGroundingBatch:
        queued_batches = max(1, self.config.semvid.prefetch_depth + 1)
        return self.grounder.prepare_batch(
            [task.request for task in tasks],
            pin_memory=self.config.semvid.preprocess_workers == 1,
            # Per-batch cap guarantees all concurrently queued batches remain
            # below the configured aggregate pinned-memory budget.
            pinned_memory_limit_bytes=self.config.semvid.pinned_memory_limit_bytes // queued_batches,
        )

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        return "out of memory" in str(error).lower() and "cuda" in str(error).lower()

    def _ground_with_oom_fallback(
        self, tasks: Sequence[_ComponentTask], prepared: PreparedGroundingBatch,
    ) -> list[GroundingPrediction | Exception]:
        try:
            return self.grounder.ground_prepared(prepared)
        except RuntimeError as error:
            if len(tasks) == 1 or not self._is_cuda_oom(error):
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            outputs: list[GroundingPrediction | Exception] = []
            for task in tasks:
                try:
                    result = self.grounder.ground_batch([task.request])[0]
                    if isinstance(result, GroundingPrediction):
                        result.telemetry["qwen_oom_fallback"] = True
                    outputs.append(result)
                except Exception as fallback_error:  # isolate retries by routed component
                    outputs.append(fallback_error)
            return outputs

    def _record_component_result(
        self, task: _ComponentTask, result: GroundingPrediction | Exception,
    ) -> None:
        context = task.context
        if isinstance(result, Exception):
            error_record: dict[str, Any] = {
                "component": asdict(task.request.component), "error": str(result),
            }
            if isinstance(result, GroundingOutputError):
                error_record.update({
                    "raw_text": result.raw_text,
                    "semvid_stats": result.semvid_stats,
                    "token_roles": result.token_roles,
                    "telemetry": result.telemetry,
                })
            if isinstance(result, EventAbsentError):
                error_record.update({
                    "event_present": False,
                    "presence_score": result.confidence,
                })
            context.proposal_errors.append(error_record)
            return
        quality: dict[str, float] = {
            "score": 0.0, "boundary_contrast": 0.0,
            "start_contrast": 0.0, "end_contrast": 0.0, "tightness": 0.0,
            "start_confidence": 0.0, "end_confidence": 0.0, "boundary_confidence": 0.0,
        }
        if context.index is not None and context.query_embedding is not None:
            quality = interval_boundary_quality(
                context.index, context.query_embedding, result.interval,
                task.request.component, self.config.proposal,
            )
        latency = float(result.telemetry.get("component_seconds", 0.0))
        context.proposals.append((quality["score"], result, quality, latency))

    def _ground_contexts(self, contexts: Sequence[_SampleContext]) -> None:
        batches = self._component_batches(contexts)
        if not batches:
            return

        def consume(tasks: Sequence[_ComponentTask], prepared: PreparedGroundingBatch) -> None:
            try:
                results = self._ground_with_oom_fallback(tasks, prepared)
            except Exception as error:
                results = [error] * len(tasks)
            if len(results) != len(tasks):
                results = [RuntimeError(
                    f"Qwen returned {len(results)} rows for a batch of {len(tasks)} requests"
                )] * len(tasks)
            for task, result in zip(tasks, results):
                self._record_component_result(task, result)

        def reject(tasks: Sequence[_ComponentTask], error: Exception) -> None:
            for task in tasks:
                self._record_component_result(task, error)

        if self.config.semvid.preprocess_workers == 0:
            for tasks in batches:
                try:
                    prepared = self._prepare_tasks(tasks)
                except Exception as error:
                    reject(tasks, error)
                else:
                    consume(tasks, prepared)
            return

        depth = max(1, self.config.semvid.prefetch_depth)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-preprocess") as executor:
            futures: dict[int, Future[PreparedGroundingBatch]] = {}
            next_to_submit = 0
            while next_to_submit < min(depth, len(batches)):
                futures[next_to_submit] = executor.submit(self._prepare_tasks, batches[next_to_submit])
                next_to_submit += 1
            for batch_index, tasks in enumerate(batches):
                future = futures.pop(batch_index)
                if next_to_submit < len(batches):
                    futures[next_to_submit] = executor.submit(self._prepare_tasks, batches[next_to_submit])
                    next_to_submit += 1
                try:
                    prepared = future.result()
                except Exception as error:
                    reject(tasks, error)
                else:
                    consume(tasks, prepared)

    def _finalize(self, context: _SampleContext, peak_memory: dict[str, int]) -> dict[str, Any]:
        if not context.proposals:
            raise RuntimeError(f"all routed component predictions failed: {context.proposal_errors}")
        context.proposals.sort(key=_proposal_sort_key)
        _, prediction, selected_quality, _ = context.proposals[0]
        sample = context.sample
        temporal_route = context.temporal_route
        assert temporal_route is not None

        decision = decide_refinement(
            prediction, selected_quality, self.config.refinement,
            expert_fps=self.config.semvid.fps,
            low_confidence_route=temporal_route.low_confidence_fallback,
        )
        refinement = None
        refinement_seconds = 0.0
        final_interval = prediction.interval
        if decision.refine:
            if context.query_embedding is None:
                context.query_embedding = self._encoder().encode_text(sample.query)
            stage_started = perf_counter()
            refinement_config = replace(self.config.refinement, fps=decision.fps)
            refinement = refine_prediction(
                sample, prediction, self._encoder(), context.query_embedding, refinement_config,
            )
            refinement_seconds = perf_counter() - stage_started
            final_interval = refinement.interval

        prediction_value = prediction.to_dict()
        prediction_value["coarse_interval"] = list(prediction.interval)
        prediction_value["interval"] = list(final_interval)
        proposals = context.proposals
        expert_attempts = [
            (value.semvid_stats, value.telemetry) for _, value, _, _ in proposals
        ] + [
            (error.get("semvid_stats") or {}, error.get("telemetry") or {})
            for error in context.proposal_errors
            if error.get("telemetry")
        ]
        index = context.index
        expert_frames = sum(int(telemetry.get("decoded_frames", 0)) for _, telemetry in expert_attempts)
        expert_pixels = sum(int(telemetry.get("decoded_pixels", 0)) for _, telemetry in expert_attempts)
        expert_vision = sum(float(
            telemetry.get("vision_encoder_seconds", 0.0)
        ) for _, telemetry in expert_attempts)
        coarse_frames = len(index.timestamps) if index is not None else 0
        coarse_decoded_frames = coarse_frames if index is not None and not context.cache_hit else 0
        coarse_decoded_pixels = index.decoded_pixels if index is not None and not context.cache_hit else 0
        coarse_vision = index.vision_encoder_seconds if index is not None and not context.cache_hit else 0.0
        refinement_frames = refinement.decoded_frames if refinement else 0
        refinement_pixels = refinement.decoded_pixels if refinement else 0
        refinement_vision = refinement.vision_encoder_seconds if refinement else 0.0
        component_latencies = [
            float(telemetry.get("component_seconds", 0.0)) for _, telemetry in expert_attempts
        ]
        ground_seconds = sum(component_latencies)

        return {
            "id": sample.id,
            "video": sample.video,
            "query": sample.query,
            "duration": sample.duration,
            "targets": [list(value) for value in sample.targets],
            "prediction": prediction_value,
            "component_predictions": [
                {
                    **value.to_dict(), "rerank_score": score,
                    "rerank_signals": quality, "latency_seconds": latency,
                }
                for score, value, quality, latency in proposals
            ],
            "component_errors": context.proposal_errors,
            "route": {
                **asdict(temporal_route),
                "retained_fraction": temporal_route.retained_union_seconds / sample.duration,
                "candidates": [
                    asdict(context.candidates[index]) for index in temporal_route.selected_candidates
                ],
            },
            "refinement": asdict(refinement) if refinement else None,
            "refinement_decision": asdict(decision),
            "cache_hit": context.cache_hit,
            "efficiency": {
                "coarse_frames": coarse_frames,
                "coarse_decoded_frames": coarse_decoded_frames,
                "coarse_decoded_pixels": coarse_decoded_pixels,
                "coarse_processor_seconds": (
                    index.processor_seconds if index is not None and not context.cache_hit else 0.0
                ),
                "coarse_vision_encoder_seconds": coarse_vision,
                "effective_coarse_fps": index.fps if index is not None else 0.0,
                "expert_seconds": sum(component.duration for component in context.ordered_components),
                "semvid_original_tokens": sum(
                    int(stats.get("orig_video_tokens", 0)) for stats, _ in expert_attempts
                ),
                "semvid_retained_tokens": sum(
                    int(stats.get("kept_video_tokens", 0)) for stats, _ in expert_attempts
                ),
                "expert_decoded_frames": expert_frames,
                "expert_decoded_pixels": expert_pixels,
                "expert_vision_encoder_seconds": expert_vision,
                "llm_prefill_tokens_before_pruning": sum(
                    int(telemetry.get("prefill_tokens_before_pruning", 0))
                    for _, telemetry in expert_attempts
                ),
                "llm_prefill_tokens_after_pruning": sum(
                    int(telemetry.get("prefill_tokens_after_pruning", 0))
                    for _, telemetry in expert_attempts
                ),
                "llm_batch_padding_tokens": sum(
                    int(telemetry.get("batch_padding_tokens", 0)) for _, telemetry in expert_attempts
                ),
                "qwen_batch_sizes": [
                    int(telemetry.get("qwen_batch_size", 1)) for _, telemetry in expert_attempts
                ],
                "qwen_oom_fallbacks": sum(
                    bool(telemetry.get("qwen_oom_fallback", False))
                    for _, telemetry in expert_attempts
                ),
                "qwen_queue_wait_seconds": sum(
                    float(telemetry.get("queue_wait_seconds", 0.0))
                    for _, telemetry in expert_attempts
                ),
                "qwen_host_to_device_seconds": sum(
                    float(telemetry.get("host_to_device_seconds", 0.0))
                    for _, telemetry in expert_attempts
                ),
                "qwen_pinned_memory_bytes": max(
                    (int(telemetry.get("pinned_memory_bytes", 0)) for _, telemetry in expert_attempts),
                    default=0,
                ),
                "component_latency_seconds": component_latencies,
                "refinement_decoded_frames": refinement_frames,
                "refinement_decoded_pixels": refinement_pixels,
                "refinement_vision_encoder_seconds": refinement_vision,
                "total_decoded_frames": coarse_decoded_frames + expert_frames + refinement_frames,
                "total_decoded_pixels": coarse_decoded_pixels + expert_pixels + refinement_pixels,
                "total_vision_encoder_seconds": coarse_vision + expert_vision + refinement_vision,
                "timing_seconds": {
                    "index": context.index_seconds,
                    "route": context.route_seconds,
                    "ground": ground_seconds,
                    "refine": refinement_seconds,
                    "total": perf_counter() - context.started,
                },
                "peak_gpu_memory_bytes": self._peak_memory() or peak_memory,
            },
        }

    def iter_results(
        self, samples: Sequence[Sample],
    ) -> Iterator[tuple[Sample, dict[str, Any] | None, Exception | None]]:
        """Run samples in bounded look-ahead groups while preserving input order."""
        lookahead = self.config.semvid.pairing_lookahead
        for start in range(0, len(samples), lookahead):
            chunk = samples[start:start + lookahead]
            self._reset_peak_memory()
            contexts: dict[str, _SampleContext] = {}
            failures: dict[str, Exception] = {}
            for sample in chunk:
                try:
                    contexts[sample.id] = self._route_sample(sample)
                except Exception as error:
                    failures[sample.id] = error
            self._ground_contexts(list(contexts.values()))
            peak_memory = self._peak_memory()
            for sample in chunk:
                if sample.id in failures:
                    yield sample, None, failures[sample.id]
                    continue
                try:
                    record = self._finalize(contexts[sample.id], peak_memory)
                except Exception as error:
                    yield sample, None, error
                else:
                    yield sample, record, None

    def run_sample(self, sample: Sample) -> dict[str, Any]:
        _, record, error = next(self.iter_results([sample]))
        if error is not None:
            raise error
        assert record is not None
        return record

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
