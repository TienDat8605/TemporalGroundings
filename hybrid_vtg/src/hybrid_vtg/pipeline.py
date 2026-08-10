"""Full-video orchestration for spatial-token grounding benchmarks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Sequence

from .config import PipelineConfig
from .semvid_bridge import (
    GroundingRequest, PreparedGroundingBatch, SemVIDGrounder,
)
from .types import Component, GroundingPrediction, Sample


@dataclass(frozen=True)
class _GroundingTask:
    sample: Sample
    request: GroundingRequest


class HybridVTGPipeline:
    """One continuous video, one spatial policy, and one generation per sample."""

    def __init__(self, config: PipelineConfig, semvid_root: Path | None = None) -> None:
        self.config = config
        self.grounder = SemVIDGrounder(
            config.grounder,
            config.spatial_allocator,
            semvid_root,
            config.observation,
        )

    @staticmethod
    def _task(sample: Sample) -> _GroundingTask:
        sample.validate()
        component = Component(0.0, sample.duration, 1.0)
        return _GroundingTask(sample, GroundingRequest(sample, component))

    def _batches(self, samples: Sequence[Sample]) -> list[list[_GroundingTask]]:
        tasks = [self._task(sample) for sample in samples]
        tasks.sort(key=lambda task: (task.sample.duration, task.sample.id))
        size = self.config.grounder.batch_size
        return [tasks[start:start + size] for start in range(0, len(tasks), size)]

    def _prepare(self, tasks: Sequence[_GroundingTask]) -> PreparedGroundingBatch:
        queued = max(1, self.config.grounder.prefetch_depth + 1)
        return self.grounder.prepare_batch(
            [task.request for task in tasks],
            pin_memory=self.config.grounder.preprocess_workers == 1,
            pinned_memory_limit_bytes=self.config.grounder.pinned_memory_limit_bytes // queued,
        )

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        text = str(error).lower()
        return "out of memory" in text and "cuda" in text

    def _infer_prepared(
        self,
        tasks: Sequence[_GroundingTask],
        prepared: PreparedGroundingBatch,
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
            results: list[GroundingPrediction | Exception] = []
            for task in tasks:
                try:
                    result = self.grounder.ground_batch([task.request])[0]
                    if isinstance(result, GroundingPrediction):
                        result.telemetry["qwen_oom_fallback"] = True
                    results.append(result)
                except Exception as fallback_error:
                    results.append(fallback_error)
            return results

    def _infer_batches(
        self, batches: Sequence[Sequence[_GroundingTask]],
    ) -> dict[str, GroundingPrediction | Exception]:
        outcomes: dict[str, GroundingPrediction | Exception] = {}

        def consume(tasks: Sequence[_GroundingTask], prepared: PreparedGroundingBatch) -> None:
            try:
                results = self._infer_prepared(tasks, prepared)
            except Exception as error:
                results = [error] * len(tasks)
            if len(results) != len(tasks):
                results = [RuntimeError(
                    f"Qwen returned {len(results)} rows for {len(tasks)} requests"
                )] * len(tasks)
            outcomes.update((task.sample.id, result) for task, result in zip(tasks, results))

        def reject(tasks: Sequence[_GroundingTask], error: Exception) -> None:
            outcomes.update((task.sample.id, error) for task in tasks)

        if self.config.grounder.preprocess_workers == 0:
            for tasks in batches:
                try:
                    prepared = self._prepare(tasks)
                except Exception as error:
                    reject(tasks, error)
                else:
                    consume(tasks, prepared)
            return outcomes

        depth = max(1, self.config.grounder.prefetch_depth)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-preprocess") as executor:
            futures: dict[int, Future[PreparedGroundingBatch]] = {}
            next_batch = 0
            while next_batch < min(depth, len(batches)):
                futures[next_batch] = executor.submit(self._prepare, batches[next_batch])
                next_batch += 1
            for batch_index, tasks in enumerate(batches):
                future = futures.pop(batch_index)
                if next_batch < len(batches):
                    futures[next_batch] = executor.submit(self._prepare, batches[next_batch])
                    next_batch += 1
                try:
                    prepared = future.result()
                except Exception as error:
                    reject(tasks, error)
                else:
                    consume(tasks, prepared)
        return outcomes

    def _record(
        self,
        sample: Sample,
        prediction: GroundingPrediction,
        elapsed_seconds: float,
        peak_memory: dict[str, int],
    ) -> dict[str, Any]:
        stats, telemetry = prediction.spatial_stats, prediction.telemetry
        original = int(stats.get("orig_video_tokens", stats.get("original_visual_tokens", 0)))
        retained = int(stats.get("kept_video_tokens", stats.get("actual_retained_tokens", 0)))
        allocator_timing = {
            key: float(telemetry.get(key, 0.0))
            for key in (
                "query_allocation_seconds", "motion_allocation_seconds",
                "boundary_allocation_seconds", "selection_seconds",
            )
        }
        return {
            "id": sample.id,
            "video": sample.video,
            "query": sample.query,
            "duration": sample.duration,
            "targets": [list(value) for value in sample.targets],
            "group": sample.group,
            "cardinality": sample.cardinality,
            "spatial_policy": self.config.spatial_allocator.spatial_policy,
            "observation_policy": self.config.observation.policy,
            "prediction": prediction.to_dict(),
            "efficiency": {
                "target_retained_tokens": int(telemetry.get("target_retained_tokens", retained)),
                "actual_retained_tokens": retained,
                "original_visual_tokens": original,
                "effective_retention_ratio": retained / original if original else 0.0,
                "per_frame_allocation": telemetry.get("per_frame_allocation"),
                "token_role_counts": prediction.token_roles,
                "selected_boundary_bands": telemetry.get("selected_boundary_bands"),
                "tpsa_audit": {
                    key: telemetry.get(key)
                    for key in (
                        "query_only_overlap_tokens", "query_only_overlap_fraction",
                        "query_only_nonprototype_overlap_tokens",
                        "query_only_nonprototype_overlap_fraction", "attempted_replacements",
                        "actual_replacements", "attempted_motion_replacements",
                        "actual_motion_replacements", "attempted_boundary_replacements",
                        "actual_boundary_replacements", "motion_gated_frames",
                        "motion_gated_frame_count", "rejected_boundary_bands",
                        "boundary_evidence", "auxiliary_quota", "quota_returned_to_query",
                    )
                },
                "hmve": telemetry.get("hmve"),
                "allocator_timing_seconds": allocator_timing,
                "decoded_frames": int(telemetry.get("decoded_frames", 0)),
                "decoded_pixels": int(telemetry.get("decoded_pixels", 0)),
                "vision_encoder_seconds": float(telemetry.get("vision_encoder_seconds", 0.0)),
                "generation_seconds": float(telemetry.get("generation_seconds", 0.0)),
                "prefill_tokens_before_pruning": int(
                    telemetry.get("prefill_tokens_before_pruning", 0)
                ),
                "prefill_tokens_after_pruning": int(
                    telemetry.get("prefill_tokens_after_pruning", 0)
                ),
                "batch_padding_tokens": int(telemetry.get("batch_padding_tokens", 0)),
                "qwen_batch_size": int(telemetry.get("qwen_batch_size", 1)),
                "model_gpu_memory_ratio": float(telemetry.get(
                    "model_gpu_memory_ratio", self.config.grounder.model_gpu_memory_ratio,
                )),
                "qwen_oom_fallback": bool(telemetry.get("qwen_oom_fallback", False)),
                "queue_wait_seconds": float(telemetry.get("queue_wait_seconds", 0.0)),
                "host_to_device_seconds": float(telemetry.get("host_to_device_seconds", 0.0)),
                "pinned_memory_bytes": int(telemetry.get("pinned_memory_bytes", 0)),
                "total_seconds": elapsed_seconds,
                "peak_gpu_memory_bytes": peak_memory,
            },
        }

    def iter_results(
        self, samples: Sequence[Sample],
    ) -> Iterator[tuple[Sample, dict[str, Any] | None, Exception | None]]:
        if len({sample.id for sample in samples}) != len(samples):
            raise ValueError("sample IDs must be unique")
        lookahead = self.config.grounder.pairing_lookahead
        for start in range(0, len(samples), lookahead):
            chunk = samples[start:start + lookahead]
            self._reset_peak_memory()
            started = perf_counter()
            outcomes = self._infer_batches(self._batches(chunk))
            peak_memory = self._peak_memory()
            elapsed = (perf_counter() - started) / max(len(chunk), 1)
            for sample in chunk:
                outcome = outcomes[sample.id]
                if isinstance(outcome, Exception):
                    yield sample, None, outcome
                else:
                    yield sample, self._record(sample, outcome, elapsed, peak_memory), None

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
