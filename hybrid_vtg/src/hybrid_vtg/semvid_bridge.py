"""Focused adapter from routed clips to the official SemVID Qwen implementation."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .config import GrounderConfig, ObservationConfig, SpatialAllocatorConfig
from .hmve import (
    EvidenceUnit, estimate_vision_transformer_tflops, evidence_unit,
    propose_corridors, query_token_embeddings, select_evidence,
)
from .timestamps import (
    consolidate_intervals,
    normalize_timestamp,
    parse_intervals_detailed,
    parse_timestamp,
)
from .types import Component, GroundingPrediction, Sample


HANDLER = "modeling_qwen3_vl_semvid.Qwen3VLForConditionalGenerationSemVID"
TPSA_HANDLER = "modeling_qwen3_vl_tpsa.Qwen3VLForConditionalGenerationTPSA"


class GroundingOutputError(ValueError):
    """Failed model output with the compute telemetry needed for accounting."""

    def __init__(
        self, message: str, raw_text: str, spatial_stats: dict[str, Any],
        token_roles: dict[str, int], telemetry: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.spatial_stats = spatial_stats
        self.semvid_stats = spatial_stats
        self.token_roles = token_roles
        self.telemetry = telemetry


def default_semvid_root() -> Path:
    override = os.environ.get("SEMVID_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[3] / "SemVID"


def validate_semvid_root(root: Path) -> None:
    required = (
        root / "src/utils/model_utils.py",
        root / "src/models/models/modeling_qwen3_vl_semvid.py",
        root / "LICENSE",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "SemVID submodule is unavailable. Run `git submodule update --init --recursive`. "
            f"Missing: {', '.join(missing)}"
        )


def _activate_upstream(root: Path) -> None:
    validate_semvid_root(root)
    root_text = str(root)
    loaded = sys.modules.get("src")
    if loaded is not None:
        locations = [str(path) for path in getattr(loaded, "__path__", [])]
        if locations and not any(path.startswith(root_text) for path in locations):
            raise RuntimeError("another top-level Python package named 'src' was imported before SemVID")
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _token_role_counts(
    model: Any, batch_index: int = 0, allocator_stats: dict[str, Any] | None = None,
) -> dict[str, int]:
    if allocator_stats:
        return {
            "prototype": int(allocator_stats.get("prototype_tokens", 0)),
            "start_boundary": int(allocator_stats.get("start_boundary_tokens", 0)),
            "end_boundary": int(allocator_stats.get("end_boundary_tokens", 0)),
            "adaptive": int(allocator_stats.get("adaptive_tokens", 0)),
        }
    coordinates = getattr(model, "last_semantic_prune_coords", None) or {}
    output = {}
    for role in ("context", "object", "motion"):
        values = coordinates.get(role, [])
        value = values[batch_index] if batch_index < len(values) else None
        output[role] = int(value.numel() // 2) if hasattr(value, "numel") else 0
    return output


def _per_frame_allocation(
    model: Any,
    batch_index: int,
    frames: int,
    patches: int,
    allocator_stats: dict[str, Any],
    spatial_policy: str,
) -> list[int] | None:
    if allocator_stats.get("per_frame_allocation") is not None:
        return [int(value) for value in allocator_stats["per_frame_allocation"]]
    if frames <= 0 or patches <= 0:
        return None
    if spatial_policy == "dense":
        return [patches] * frames
    coordinates = getattr(model, "last_semantic_prune_coords", None) or {}
    values = []
    for role in ("context", "object", "motion"):
        rows = coordinates.get(role, [])
        if batch_index < len(rows) and hasattr(rows[batch_index], "numel"):
            values.append(rows[batch_index])
    if not values:
        return None
    import torch
    pairs = torch.cat(values, dim=0)
    if pairs.numel() == 0:
        return [0] * frames
    flat = torch.unique(pairs[:, 0].long() * patches + pairs[:, 1].long())
    counts = torch.bincount(flat // patches, minlength=frames)
    return [int(value) for value in counts.tolist()]


def _render_generation_prompt(processor: Any, prompt: list[dict[str, Any]], force_stop_thinking: bool) -> str:
    """Render a Qwen prompt using the same thinking control as SemVID evaluation."""
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if force_stop_thinking:
        text += "</think>"
    return text


@dataclass(frozen=True)
class GroundingRequest:
    sample: Sample
    component: Component

    @property
    def estimated_visual_load(self) -> float:
        return self.component.duration


@dataclass
class PreparedGroundingBatch:
    requests: tuple[GroundingRequest, ...]
    inputs: dict[str, Any]
    prompt_length: int
    input_stats: tuple[dict[str, int], ...]
    preparation_seconds: float
    pinned_memory_bytes: int
    ready_at: float
    hmve_scout_units: tuple[Any, ...] = ()
    hmve_reference_tokens: int = 0


@dataclass(frozen=True)
class _HMVEInputUnit:
    video: Any
    metadata: Any
    pass_id: int
    absolute_time: float
    source_observation: int
    cache_index: int = -1


@dataclass(frozen=True)
class _HMVEDecodedPass:
    units: tuple[_HMVEInputUnit, ...]
    observations: tuple[_HMVEInputUnit, ...]


class _FirstLogitsCapture:
    """Read-only generation hook used only by batch-equivalence validation."""

    def __init__(self) -> None:
        self.values: Any = None

    def __call__(self, _input_ids: Any, scores: Any) -> Any:
        if self.values is None:
            self.values = scores.detach().float().cpu()
        return scores


class SemVIDGrounder:
    """Batched inference adapter for routed video components."""

    def __init__(
        self,
        config: GrounderConfig,
        spatial_allocator: SpatialAllocatorConfig | None = None,
        semvid_root: Path | None = None,
        observation: ObservationConfig | None = None,
    ) -> None:
        try:
            import torch
            from transformers import GenerationConfig
        except ImportError as error:
            raise RuntimeError("install SemVID/requirements.txt before loading the grounder") from error
        if not torch.cuda.is_available():
            raise RuntimeError("SemVID grounding requires a CUDA GPU; CPU-only routing remains available")
        root = semvid_root or default_semvid_root()
        allocator = spatial_allocator or SpatialAllocatorConfig()
        observation = observation or ObservationConfig()
        if observation.policy == "hmve" and config.batch_size != 1:
            raise ValueError("HMVE Phase A supports Qwen batch size 1 only")
        _activate_upstream(root)
        policy = allocator.spatial_policy
        model_source = root / "src/models/models/modeling_qwen3_vl_semvid.py"
        if config.batch_size > 1 and "pruned_attention_mask" not in model_source.read_text(encoding="utf-8"):
            raise RuntimeError(
                "Qwen microbatching requires the bundled SemVID cache/position patch. "
                "Run `bash hybrid_vtg/scripts/apply_semvid_patches.sh` first."
            )
        if policy.startswith("tpsa_"):
            if "merged_grid_hw" not in model_source.read_text(encoding="utf-8"):
                raise RuntimeError(
                    "TPSA requires the bundled SemVID visual-selection hook patch. "
                    "Run `bash hybrid_vtg/scripts/apply_semvid_patches.sh` first."
                )
            from . import qwen_tpsa
            sys.modules["src.models.models.modeling_qwen3_vl_tpsa"] = qwen_tpsa
        from src.utils.model_utils import load_hosted_model

        hyperparameters = {
            "enable_semantic_prune": policy != "dense",
            "semantic_retention_ratio": allocator.retention_ratio,
            "semantic_stage1_topk_segments": 0,
            "semantic_stage1_smooth_win": 1,
            "semantic_frame_weight_alpha": 0.7,
            "semantic_obj_ratio": 0.6,
            "semantic_mmr_lambda": allocator.mmr_lambda,
            "semantic_min_tokens_per_frame": 0,
            "semantic_motion_query_beta": allocator.motion_query_beta,
            "semantic_query_token_max": 4096 if policy.startswith("tpsa_") else 50,
            "ablation_uniform_allocation": policy == "uniform",
            "ablation_use_semantic_selection": "semvid",
            "spatial_policy": policy,
            "tpsa_fps": config.fps,
            "tpsa_mmr_lambda": allocator.mmr_lambda,
            "tpsa_relevance_top_fraction": allocator.relevance_top_fraction,
            "tpsa_auxiliary_fraction": allocator.auxiliary_fraction,
            "tpsa_boundary_share": allocator.boundary_share,
            "tpsa_motion_query_beta": allocator.motion_query_beta,
            "tpsa_evidence_mad_multiplier": allocator.evidence_mad_multiplier,
            "tpsa_boundary_window_seconds": allocator.boundary_window_seconds,
            "tpsa_boundary_nms_seconds": allocator.boundary_nms_seconds,
            "tpsa_boundary_expansion_seconds": allocator.boundary_expansion_seconds,
            "tpsa_maximum_boundary_bands": allocator.maximum_boundary_bands,
            "attn_implementation": config.attention,
        }
        handler = TPSA_HANDLER if policy.startswith("tpsa_") else HANDLER
        self.model, self.processor = load_hosted_model(
            config.model, model_handler=handler, model_hyper_parameters=hyperparameters,
            dtype=config.dtype, backend="vllm",
            max_memory_ratio=config.model_gpu_memory_ratio,
        )
        if self.model.config.text_config.pad_token_id is None:
            self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.eval().requires_grad_(False)
        # Transformers/Accelerate may leave checkpoint-loading cache reserved.
        # HMVE needs that space for two explicit vision passes and retained scout
        # projections, so release unused blocks before accepting the first batch.
        torch.cuda.empty_cache()
        self.config = config
        self.spatial_allocator = allocator
        self.observation = observation
        self.torch = torch
        self._processor_lock = threading.Lock()
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens, do_sample=False,
        )

    @staticmethod
    def _instruction(sample: Sample) -> str:
        if sample.cardinality == "multi":
            return (
                f"Find every disjoint time interval where this event occurs: {sample.query!r}.\n"
                f"Timestamps must be seconds within [0.000, {sample.duration:.3f}]. "
                "Return only a JSON array of numeric pairs like "
                "[[12.0, 18.0], [42.0, 49.5]], ordered by start time. "
                "Do not use objects, keys, prose, or Markdown fences. "
                "Return [] if the event never occurs, and do not omit repeated occurrences."
            )
        else:
            return (
                f"Localize the event described by this query in the video: {sample.query}\n"
                f"Return only JSON timestamps within [0.000, {sample.duration:.3f}]: "
                '{"start": number, "end": number}.'
            )

    def _video_content(
        self,
        sample: Sample,
        component: Component,
        *,
        fps: float,
        total_pixel_tokens: int,
    ) -> dict[str, Any]:
        total_pixels = total_pixel_tokens * 16 * 16 * 4
        minimum_pixels = self.config.minimum_pixel_tokens * 16 * 16 * 4
        return {
            "type": "video", "video": sample.video_path,
            "video_start": component.start, "video_end": component.end,
            "fps": fps, "total_pixels": total_pixels, "min_pixels": minimum_pixels,
            "max_frames": self.config.max_frames,
        }

    def _prompt(self, sample: Sample, component: Component) -> list[dict[str, Any]]:
        return [{
            "role": "user",
            "content": [
                self._video_content(
                    sample, component, fps=self.config.fps,
                    total_pixel_tokens=self.config.total_pixel_tokens,
                ),
                {"type": "text", "text": self._instruction(sample)},
            ],
        }]

    def _unit_prompt(self, sample: Sample, units: Sequence[_HMVEInputUnit]) -> list[dict[str, Any]]:
        content = [
            {"type": "video", "video": unit.video}
            for unit in units
        ]
        content.append({"type": "text", "text": self._instruction(sample)})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _video_stats(video: Any) -> dict[str, int]:
        shape = tuple(int(value) for value in getattr(video, "shape", ()))
        if len(shape) < 4:
            return {"decoded_frames": 0, "decoded_pixels": 0}
        return {
            "decoded_frames": shape[0],
            "decoded_pixels": shape[0] * shape[-2] * shape[-1],
        }

    @staticmethod
    def _metadata_value(metadata: Any, name: str) -> Any:
        return getattr(metadata, name) if hasattr(metadata, name) else metadata[name]

    def _split_hmve_units(
        self,
        video: Any,
        metadata: Any,
        *,
        pass_id: int,
        source_observation: int,
    ) -> list[_HMVEInputUnit]:
        from transformers.video_utils import VideoMetadata

        temporal_patch_size = max(
            int(getattr(getattr(self.model.visual, "patch_embed", None), "temporal_patch_size", 2)),
            1,
        )
        indices = list(self._metadata_value(metadata, "frames_indices"))
        fps = float(self._metadata_value(metadata, "fps"))
        if len(indices) != int(video.shape[0]):
            raise ValueError("HMVE decoded frames and absolute frame indices differ")
        units = []
        cache_index = 0
        for start in range(0, len(indices), temporal_patch_size):
            stop = min(start + temporal_patch_size, len(indices))
            frames = video[start:stop]
            unit_indices = indices[start:stop]
            if stop - start < temporal_patch_size:
                repeat = temporal_patch_size - (stop - start)
                frames = self.torch.cat((frames, frames[-1:].repeat(repeat, 1, 1, 1)), dim=0)
                unit_indices.extend([unit_indices[-1]] * repeat)
            unit_metadata = VideoMetadata(
                total_num_frames=int(self._metadata_value(metadata, "total_num_frames")),
                fps=fps,
                width=int(frames.shape[-1]),
                height=int(frames.shape[-2]),
                duration=float(self._metadata_value(metadata, "total_num_frames")) / fps,
                video_backend=str(self._metadata_value(metadata, "video_backend")),
                frames_indices=unit_indices,
            )
            units.append(_HMVEInputUnit(
                video=frames,
                metadata=unit_metadata,
                pass_id=pass_id,
                absolute_time=sum(unit_indices) / len(unit_indices) / fps,
                source_observation=source_observation,
                cache_index=cache_index,
            ))
            cache_index += 1
        return units

    def _hmve_observation(
        self,
        video: Any,
        metadata: Any,
        *,
        pass_id: int,
        source_observation: int,
    ) -> _HMVEInputUnit:
        indices = list(self._metadata_value(metadata, "frames_indices"))
        fps = float(self._metadata_value(metadata, "fps"))
        if not indices or fps <= 0:
            raise ValueError("HMVE observation metadata requires frame indices and FPS")
        return _HMVEInputUnit(
            video=video,
            metadata=metadata,
            pass_id=pass_id,
            absolute_time=sum(indices) / len(indices) / fps,
            source_observation=source_observation,
            cache_index=0,
        )

    def _split_hmve_embeddings(
        self,
        observation_embeddings: Sequence[Any],
        observation_grids: Any,
        units: Sequence[_HMVEInputUnit],
    ) -> tuple[Any, ...]:
        """Split grouped encoder outputs into timestamped temporal units."""
        if len(observation_embeddings) != int(observation_grids.shape[0]):
            raise RuntimeError("HMVE encoder output/grid count mismatch")
        lookup: dict[tuple[int, int], Any] = {}
        merge = int(self.model.visual.spatial_merge_size)
        for source, (embeddings, grid) in enumerate(zip(
            observation_embeddings, observation_grids,
        )):
            temporal = int(grid[0].item())
            related = sorted(
                (unit for unit in units if unit.source_observation == source),
                key=lambda unit: unit.cache_index,
            )
            if len(related) != temporal:
                raise RuntimeError(
                    f"HMVE observation {source} encoded {temporal} temporal units, "
                    f"but provenance contains {len(related)}"
                )
            tokens_per_unit = (
                (int(grid[1].item()) // merge) * (int(grid[2].item()) // merge)
            )
            if int(embeddings.shape[0]) != temporal * tokens_per_unit:
                raise RuntimeError("HMVE grouped projected-token accounting mismatch")
            for unit, value in zip(related, embeddings.split(tokens_per_unit, dim=0)):
                lookup[(source, unit.cache_index)] = value
        return tuple(
            lookup[(unit.source_observation, unit.cache_index)] for unit in units
        )

    def _process_hmve_units(
        self,
        sample: Sample,
        units: Sequence[_HMVEInputUnit],
    ) -> dict[str, Any]:
        prompt = self._unit_prompt(sample, units)
        text = _render_generation_prompt(
            self.processor, prompt, self.config.force_stop_thinking,
        )
        inputs = self.processor(
            text=[text],
            images=None,
            videos=[unit.video for unit in units],
            video_metadata=[unit.metadata for unit in units],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            do_resize=False,
            do_sample_frames=False,
        )
        return dict(inputs)

    def _reference_visual_tokens(self, sample: Sample) -> int:
        """Estimate the dense one-pass Qwen token count without decoding the timeline."""
        import decord
        from qwen_vl_utils.vision_process import (
            FRAME_FACTOR, VIDEO_MAX_TOKEN_NUM, smart_nframes, smart_resize,
        )

        reader = decord.VideoReader(sample.video_path, num_threads=1)
        total_frames = len(reader)
        source_fps = float(reader.get_avg_fps())
        height, width = [int(value) for value in reader[0].shape[:2]]
        frames = smart_nframes(
            {"fps": self.config.fps, "max_frames": self.config.max_frames},
            total_frames=total_frames,
            video_fps=source_fps,
        )
        image_factor = 16 * int(self.model.visual.spatial_merge_size)
        minimum_pixels = self.config.minimum_pixel_tokens * image_factor * image_factor
        total_pixels = self.config.total_pixel_tokens * image_factor * image_factor
        maximum_pixels = max(
            min(VIDEO_MAX_TOKEN_NUM * image_factor * image_factor, total_pixels / frames * FRAME_FACTOR),
            int(minimum_pixels * 1.05),
        )
        resized_height, resized_width = smart_resize(
            height, width, factor=image_factor,
            min_pixels=minimum_pixels, max_pixels=maximum_pixels,
        )
        temporal_patch_size = max(
            int(getattr(getattr(self.model.visual, "patch_embed", None), "temporal_patch_size", 2)),
            1,
        )
        temporal = frames // temporal_patch_size
        merged_height = resized_height // image_factor
        merged_width = resized_width // image_factor
        return temporal * merged_height * merged_width

    def _prepare_hmve_batch(
        self,
        requests: Sequence[GroundingRequest],
        *,
        pin_memory: bool,
        pinned_memory_limit_bytes: int | None,
        started: float,
    ) -> PreparedGroundingBatch:
        if len(requests) != 1:
            raise ValueError("HMVE Phase A supports one request at a time")
        try:
            import qwen_vl_utils
        except ImportError as error:
            raise RuntimeError("qwen-vl-utils is required by HMVE") from error
        request = requests[0]
        scout_prompt = [{
            "role": "user",
            "content": [
                self._video_content(
                    request.sample,
                    request.component,
                    fps=self.observation.scout_fps,
                    total_pixel_tokens=self.observation.scout_total_pixel_tokens,
                ),
                {"type": "text", "text": self._instruction(request.sample)},
            ],
        }]
        _, decoded, _ = qwen_vl_utils.process_vision_info(
            scout_prompt,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if not decoded or len(decoded) != 1:
            raise RuntimeError("HMVE scout must decode exactly one full-timeline observation")
        scout_video, scout_metadata = decoded[0]
        scout_units = self._split_hmve_units(
            scout_video,
            scout_metadata,
            pass_id=0,
            source_observation=0,
        )
        scout_observation = self._hmve_observation(
            scout_video,
            scout_metadata,
            pass_id=0,
            source_observation=0,
        )
        with self._processor_lock:
            inputs = self._process_hmve_units(request.sample, [scout_observation])
            query_tokens = self.processor.tokenizer(
                [request.sample.query], padding=True, truncation=True,
                return_tensors="pt", add_special_tokens=True,
            )
        inputs["query_ids"] = query_tokens["input_ids"]
        inputs["query_attention_mask"] = query_tokens.get("attention_mask")
        pinned_memory_bytes = 0
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for value in inputs.values()
            if self.torch.is_tensor(value) and value.device.type == "cpu"
        )
        should_pin = pin_memory and (
            pinned_memory_limit_bytes is None or tensor_bytes <= pinned_memory_limit_bytes
        )
        if should_pin:
            for key, value in tuple(inputs.items()):
                if self.torch.is_tensor(value) and value.device.type == "cpu":
                    pinned_memory_bytes += value.numel() * value.element_size()
                    inputs[key] = value.pin_memory()
        return PreparedGroundingBatch(
            requests=tuple(requests),
            inputs=inputs,
            prompt_length=int(inputs["input_ids"].shape[-1]),
            input_stats=(self._video_stats(scout_video),),
            preparation_seconds=perf_counter() - started,
            pinned_memory_bytes=pinned_memory_bytes,
            ready_at=perf_counter(),
            hmve_scout_units=tuple(scout_units),
            hmve_reference_tokens=self._reference_visual_tokens(request.sample),
        )

    def prepare_batch(
        self, requests: Sequence[GroundingRequest], *, pin_memory: bool = False,
        pinned_memory_limit_bytes: int | None = None,
    ) -> PreparedGroundingBatch:
        """Decode and tokenize one microbatch without moving tensors to CUDA."""
        if not requests:
            raise ValueError("cannot prepare an empty grounding batch")
        if len(requests) > self.config.batch_size:
            raise ValueError(f"grounding batch exceeds configured size {self.config.batch_size}")
        started = perf_counter()
        if self.observation.policy == "hmve":
            return self._prepare_hmve_batch(
                requests,
                pin_memory=pin_memory,
                pinned_memory_limit_bytes=pinned_memory_limit_bytes,
                started=started,
            )
        try:
            import qwen_vl_utils
        except ImportError as error:
            raise RuntimeError("qwen-vl-utils is required by SemVID") from error
        prompts = [self._prompt(request.sample, request.component) for request in requests]
        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
            prompts, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
        )
        metadata = None
        if video_inputs is not None:
            video_inputs, metadata = zip(*video_inputs)
            video_inputs, metadata = list(video_inputs), list(metadata)
        if len(video_inputs or []) != len(requests):
            raise RuntimeError("each grounding request must produce exactly one decoded video")
        input_stats = tuple(self._video_stats(video) for video in video_inputs or [])
        if isinstance(video_kwargs, (list, tuple)):
            video_kwargs = video_kwargs[0] if video_kwargs else {}
        processor_kwargs = {"do_sample_frames": video_kwargs.get("do_sample_frames", False)}
        with self._processor_lock:
            texts = [
                _render_generation_prompt(self.processor, prompt, self.config.force_stop_thinking)
                for prompt in prompts
            ]
            inputs = self.processor(
                text=texts, images=image_inputs, videos=video_inputs, video_metadata=metadata,
                return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False,
                do_resize=False, **processor_kwargs,
            )
            query_tokens = self.processor.tokenizer(
                [request.sample.query for request in requests], padding=True, truncation=True,
                return_tensors="pt", add_special_tokens=True,
            )
        inputs = dict(inputs)
        inputs["query_ids"] = query_tokens["input_ids"]
        inputs["query_attention_mask"] = query_tokens.get("attention_mask")
        pinned_memory_bytes = 0
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for value in inputs.values()
            if self.torch.is_tensor(value) and value.device.type == "cpu"
        )
        should_pin = pin_memory and (
            pinned_memory_limit_bytes is None or tensor_bytes <= pinned_memory_limit_bytes
        )
        if should_pin:
            for key, value in tuple(inputs.items()):
                if self.torch.is_tensor(value) and value.device.type == "cpu":
                    pinned_memory_bytes += value.numel() * value.element_size()
                    inputs[key] = value.pin_memory()
        return PreparedGroundingBatch(
            requests=tuple(requests), inputs=inputs,
            prompt_length=int(inputs["input_ids"].shape[-1]),
            input_stats=input_stats,
            preparation_seconds=perf_counter() - started,
            pinned_memory_bytes=pinned_memory_bytes,
            ready_at=perf_counter(),
        )

    def _decode_hmve_corridors(
        self,
        request: GroundingRequest,
        corridors: Sequence[Any],
    ) -> _HMVEDecodedPass:
        try:
            import qwen_vl_utils
        except ImportError as error:
            raise RuntimeError("qwen-vl-utils is required by HMVE") from error
        total_duration = sum(float(value.end - value.start) for value in corridors)
        content = []
        for corridor in corridors:
            fraction = (corridor.end - corridor.start) / max(total_duration, 1e-6)
            pixel_tokens = max(
                self.config.minimum_pixel_tokens,
                int(round(self.config.total_pixel_tokens * fraction)),
            )
            item = self._video_content(
                request.sample,
                Component(corridor.start, corridor.end, corridor.score),
                fps=self.config.fps,
                total_pixel_tokens=pixel_tokens,
            )
            item["max_frames"] = max(2, int(round(self.config.max_frames * fraction)))
            content.append(item)
        content.append({"type": "text", "text": self._instruction(request.sample)})
        prompt = [{"role": "user", "content": content}]
        _, decoded, _ = qwen_vl_utils.process_vision_info(
            prompt,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if not decoded or len(decoded) != len(corridors):
            raise RuntimeError("HMVE detailed pass did not decode every selected corridor")
        units = []
        observations = []
        for observation_index, (video, metadata) in enumerate(decoded):
            observations.append(self._hmve_observation(
                video,
                metadata,
                pass_id=1,
                source_observation=observation_index,
            ))
            units.extend(self._split_hmve_units(
                video,
                metadata,
                pass_id=1,
                source_observation=observation_index,
            ))
        return _HMVEDecodedPass(tuple(units), tuple(observations))

    def _video_unit_positions(
        self,
        input_ids: Any,
        attention_mask: Any,
    ) -> list[Any]:
        ids = input_ids[0]
        valid = attention_mask[0].bool()
        starts = self.torch.nonzero(
            (ids == self.model.config.vision_start_token_id) & valid,
            as_tuple=False,
        ).flatten().tolist()
        ends = self.torch.nonzero(
            (ids == self.model.config.vision_end_token_id) & valid,
            as_tuple=False,
        ).flatten().tolist()
        if len(starts) != len(ends):
            raise ValueError("HMVE vision marker count mismatch")
        positions = []
        for start, end in zip(starts, ends):
            span = self.torch.arange(start + 1, end, device=ids.device)
            chosen = span[ids[span] == self.model.config.video_token_id]
            if chosen.numel():
                positions.append(chosen)
        return positions

    def _hmve_generate(
        self,
        prepared: PreparedGroundingBatch,
        scout_inputs: dict[str, Any],
        generation_kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], int, float, float]:
        request = prepared.requests[0]
        device = scout_inputs["input_ids"].device
        controller_started = perf_counter()
        self.model.last_tpsa_stats_batch = []

        self.torch.cuda.synchronize()
        scout_encoder_started = perf_counter()
        scout_pixels = scout_inputs.pop("pixel_values_videos")
        scout_grids = scout_inputs["video_grid_thw"]
        scout_observation_embeds, scout_deepstack, scout_token_scores = (
            self.model.model.get_video_features(
                scout_pixels,
                scout_grids,
                return_token_scores=False,
            )
        )
        self.torch.cuda.synchronize()
        scout_encoder_seconds = perf_counter() - scout_encoder_started
        scout_embeds = self._split_hmve_embeddings(
            scout_observation_embeds,
            scout_grids,
            prepared.hmve_scout_units,
        )
        del scout_pixels, scout_observation_embeds, scout_deepstack, scout_token_scores
        self.torch.cuda.empty_cache()
        query_embeddings = query_token_embeddings(
            self.model.get_input_embeddings(),
            scout_inputs["query_ids"],
            scout_inputs.get("query_attention_mask"),
        )
        scout_strength = []
        for embeddings in scout_embeds:
            relevance = evidence_unit(
                pass_id=0,
                absolute_time=0.0,
                grid_height=1,
                grid_width=int(embeddings.shape[0]),
                source_height=1,
                source_width=int(embeddings.shape[0]),
                embeddings=embeddings,
                query_embeddings=query_embeddings,
                source_observation=0,
            ).query_relevance
            top = max(1, int(round(relevance.numel() * 0.10)))
            scout_strength.append(relevance.topk(top).values.mean())
        times = self.torch.tensor(
            [unit.absolute_time for unit in prepared.hmve_scout_units],
            device=device,
            dtype=self.torch.float32,
        )
        relevance_curve = self.torch.stack(scout_strength).float()
        detail_tokens_per_second = prepared.hmve_reference_tokens / max(request.sample.duration, 1e-6)
        target_tokens = min(
            prepared.hmve_reference_tokens,
            int(round(
                self.spatial_allocator.retention_ratio * prepared.hmve_reference_tokens
            )),
        )
        if target_tokens < len(prepared.hmve_scout_units):
            raise ValueError(
                "HMVE exact final budget is smaller than the required scout-anchor count: "
                f"{target_tokens} tokens for {len(prepared.hmve_scout_units)} units. "
                "Reduce --hmve-scout-fps or increase --retention-ratio."
            )
        minimum_detailed_seconds = (
            target_tokens * self.observation.detailed_budget_fraction
            / max(detail_tokens_per_second, 1e-6)
        )
        corridors = propose_corridors(
            times,
            relevance_curve,
            request.sample.duration,
            maximum_corridors=self.observation.maximum_corridors,
            minimum_seconds=self.observation.minimum_corridor_seconds,
            margin_seconds=self.observation.corridor_margin_seconds,
            nms_seconds=self.observation.corridor_nms_seconds,
            minimum_total_seconds=minimum_detailed_seconds,
        )
        detailed_pass = self._decode_hmve_corridors(request, corridors)
        detailed_units = list(detailed_pass.units)
        final_units = sorted(
            [*prepared.hmve_scout_units, *detailed_units],
            key=lambda value: (
                value.absolute_time, value.pass_id, value.source_observation, value.cache_index,
            ),
        )
        with self._processor_lock:
            detailed_inputs = self._process_hmve_units(
                request.sample, detailed_pass.observations,
            )
        detailed_pixels = detailed_inputs["pixel_values_videos"].to(device)
        detailed_grids = detailed_inputs["video_grid_thw"].to(device)

        self.torch.cuda.synchronize()
        detailed_encoder_started = perf_counter()
        detailed_observation_embeds, detailed_deepstack, detailed_token_scores = (
            self.model.model.get_video_features(
                detailed_pixels,
                detailed_grids,
                return_token_scores=False,
            )
        )
        self.torch.cuda.synchronize()
        detailed_encoder_seconds = perf_counter() - detailed_encoder_started
        detailed_embeds = self._split_hmve_embeddings(
            detailed_observation_embeds,
            detailed_grids,
            detailed_units,
        )
        del (
            detailed_pixels,
            detailed_inputs,
            detailed_observation_embeds,
            detailed_deepstack,
            detailed_token_scores,
        )
        self.torch.cuda.empty_cache()

        with self._processor_lock:
            final_inputs = self._process_hmve_units(request.sample, final_units)
        final_inputs.pop("pixel_values_videos", None)
        final_inputs = {
            key: value.to(device) if self.torch.is_tensor(value) else value
            for key, value in final_inputs.items()
        }
        grids = final_inputs["video_grid_thw"]
        evidence: list[EvidenceUnit] = []
        scout_by_provenance = {
            (unit.source_observation, unit.cache_index): embeddings
            for unit, embeddings in zip(prepared.hmve_scout_units, scout_embeds)
        }
        detailed_by_provenance = {
            (unit.source_observation, unit.cache_index): embeddings
            for unit, embeddings in zip(detailed_units, detailed_embeds)
        }
        for unit_index, unit in enumerate(final_units):
            provenance = (unit.source_observation, unit.cache_index)
            embeddings = (
                scout_by_provenance[provenance]
                if unit.pass_id == 0 else detailed_by_provenance[provenance]
            )
            merge = int(self.model.visual.spatial_merge_size)
            grid_height = int(grids[unit_index, 1].item()) // merge
            grid_width = int(grids[unit_index, 2].item()) // merge
            if grid_height * grid_width != int(embeddings.shape[0]):
                raise RuntimeError("HMVE projected-token grid mismatch")
            evidence.append(evidence_unit(
                pass_id=unit.pass_id,
                absolute_time=unit.absolute_time,
                grid_height=grid_height,
                grid_width=grid_width,
                source_height=int(unit.video.shape[-2]),
                source_width=int(unit.video.shape[-1]),
                embeddings=embeddings,
                query_embeddings=query_embeddings,
                source_observation=unit.source_observation,
            ))
        selection = select_evidence(
            evidence,
            target_tokens,
            deduplication_similarity=self.observation.deduplication_similarity,
        )

        input_ids = final_inputs["input_ids"]
        attention_mask = final_inputs["attention_mask"]
        unit_positions = self._video_unit_positions(input_ids, attention_mask)
        if len(unit_positions) != len(final_units):
            raise RuntimeError(
                f"HMVE prompt contains {len(unit_positions)} units, expected {len(final_units)}"
            )
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        visual_mask = self.torch.zeros_like(attention_mask, dtype=self.torch.bool)
        selected_visual_positions = []
        for unit_index, (positions, embeddings, local_indices) in enumerate(zip(
            unit_positions, [value.embeddings for value in evidence], selection.local_indices,
        )):
            if positions.numel() != embeddings.shape[0]:
                raise RuntimeError("HMVE prompt placeholders do not match projected evidence")
            inputs_embeds[0, positions] = embeddings.to(inputs_embeds.dtype)
            visual_mask[0, positions] = True
            if local_indices.numel():
                selected_visual_positions.append(positions[local_indices])
        selected_visual = self.torch.cat(selected_visual_positions).sort().values
        keep_visual = self.torch.zeros_like(attention_mask, dtype=self.torch.bool)
        keep_visual[0, selected_visual] = True
        keep = attention_mask.bool() & (~visual_mask | keep_visual)
        keep_indices = self.torch.nonzero(keep[0], as_tuple=False).flatten()
        original_positions, _ = self.model.model.get_rope_index(
            input_ids,
            image_grid_thw=None,
            video_grid_thw=grids,
            attention_mask=attention_mask,
        )
        compact_ids = input_ids[:, keep_indices]
        compact_embeds = inputs_embeds[:, keep_indices]
        compact_attention = attention_mask[:, keep_indices]
        compact_positions = original_positions[:, :, keep_indices]
        compact_length = int(compact_ids.shape[-1])
        self.model.model.rope_deltas = (
            compact_positions.max(dim=0).values.max(dim=-1, keepdim=True).values
            + 1 - compact_length
        )
        controller_wall_seconds = perf_counter() - controller_started
        controller_seconds = max(
            0.0,
            controller_wall_seconds - scout_encoder_seconds - detailed_encoder_seconds,
        )

        self.torch.cuda.synchronize()
        generation_started = perf_counter()
        generated = self.model.generate(
            input_ids=compact_ids,
            inputs_embeds=compact_embeds,
            attention_mask=compact_attention,
            position_ids=compact_positions,
            generation_config=self.generation_config,
            use_model_defaults=False,
            **generation_kwargs,
        )
        self.torch.cuda.synchronize()
        generation_seconds = perf_counter() - generation_started

        scout_frames = sum(int(unit.video.shape[0]) for unit in prepared.hmve_scout_units)
        detailed_frames = sum(int(unit.video.shape[0]) for unit in detailed_units)
        scout_pixels = sum(
            int(unit.video.shape[0] * unit.video.shape[-2] * unit.video.shape[-1])
            for unit in prepared.hmve_scout_units
        )
        detailed_pixels_count = sum(
            int(unit.video.shape[0] * unit.video.shape[-2] * unit.video.shape[-1])
            for unit in detailed_units
        )
        created_by_pass = {
            0: sum(value.token_count for value in evidence if value.pass_id == 0),
            1: sum(value.token_count for value in evidence if value.pass_id == 1),
        }
        vision_config = self.model.config.vision_config
        pass_tflops = {
            "0": estimate_vision_transformer_tflops(
                vision_config, scout_grids,
            ),
            "1": estimate_vision_transformer_tflops(vision_config, detailed_grids),
        }
        stats = {
            "orig_video_tokens": prepared.hmve_reference_tokens,
            "kept_video_tokens": target_tokens,
            "target_retained_tokens": target_tokens,
            "actual_retained_tokens": selection.actual_tokens,
            "original_visual_tokens": prepared.hmve_reference_tokens,
            "effective_retention_ratio": target_tokens / max(prepared.hmve_reference_tokens, 1),
            "orig_seq_len": int(input_ids.shape[-1]),
            "new_seq_len": compact_length,
            "per_frame_allocation": [
                int(selection.local_indices[index].numel())
                for index, value in enumerate(evidence) if value.pass_id == 0
            ],
            "prototype_tokens": selection.anchor_tokens,
            "start_boundary_tokens": 0,
            "end_boundary_tokens": 0,
            "adaptive_tokens": selection.actual_tokens - selection.anchor_tokens,
            "hmve": {
                "observation_policy": "hmve",
                "encoder_calls": 2,
                "llm_generations": 1,
                "corridors": [
                    {"start": value.start, "end": value.end, "score": value.score}
                    for value in corridors
                ],
                "passes": [
                    {
                        "pass_id": 0,
                        "decoded_frames": scout_frames,
                        "decoded_pixels": scout_pixels,
                        "created_tokens": created_by_pass[0],
                        "retained_tokens": selection.retained_by_pass.get(0, 0),
                        "encoder_seconds": scout_encoder_seconds,
                        "raw_patch_tokens": int(
                            scout_grids.prod(dim=-1).sum().item()
                        ),
                        "vision_transformer_core_tflops_estimate": pass_tflops["0"],
                    },
                    {
                        "pass_id": 1,
                        "decoded_frames": detailed_frames,
                        "decoded_pixels": detailed_pixels_count,
                        "created_tokens": created_by_pass[1],
                        "retained_tokens": selection.retained_by_pass.get(1, 0),
                        "encoder_seconds": detailed_encoder_seconds,
                        "raw_patch_tokens": int(detailed_grids.prod(dim=-1).sum().item()),
                        "vision_transformer_core_tflops_estimate": pass_tflops["1"],
                    },
                ],
                "created_tokens": sum(created_by_pass.values()),
                "retained_tokens": selection.actual_tokens,
                "global_anchor_tokens": selection.anchor_tokens,
                "redundant_coarse_tokens": selection.redundant_coarse_tokens,
                "cache_reused_scout_units": len(prepared.hmve_scout_units),
                "absolute_timestamps_preserved": True,
                "original_position_ids_preserved": True,
                "controller_seconds": controller_seconds,
                "vision_flops_estimate_kind": "full-attention transformer core; excludes patch embed and merger",
            },
            "decoded_frames": scout_frames + detailed_frames,
            "decoded_pixels": scout_pixels + detailed_pixels_count,
            "vision_encoder_seconds": scout_encoder_seconds + detailed_encoder_seconds,
        }
        self.model.fastvid_last_stats = stats
        self.model.fastvid_last_stats_batch = [stats]
        self.model.last_hmve_stats_batch = [stats]
        accounting_inputs = {
            "input_ids": compact_ids,
            "attention_mask": compact_attention,
        }
        return generated, accounting_inputs, compact_length, (
            scout_encoder_seconds + detailed_encoder_seconds
        ), generation_seconds

    def ground_prepared(
        self, prepared: PreparedGroundingBatch,
    ) -> list[GroundingPrediction | Exception]:
        """Run one prepared microbatch and isolate parsing failures by row."""
        started = perf_counter()
        device = getattr(self.model, "device", self.torch.device("cuda:0"))
        transfer_started = perf_counter()
        inputs = {
            key: value.to(device, non_blocking=value.is_pinned()) if self.torch.is_tensor(value) else value
            for key, value in prepared.inputs.items()
        }
        self.torch.cuda.synchronize()
        transfer_seconds = perf_counter() - transfer_started
        logits_capture = _FirstLogitsCapture() if self.config.capture_validation_logits else None
        generation_kwargs = {"logits_processor": [logits_capture]} if logits_capture is not None else {}
        if self.observation.policy == "hmve":
            with self.torch.inference_mode():
                generated, inputs, prompt_length, hmve_vision_seconds, generation_seconds = (
                    self._hmve_generate(prepared, inputs, generation_kwargs)
                )
            vision_seconds = [hmve_vision_seconds]
        else:
            vision_started = [0.0]
            vision_seconds = [0.0]

            def before_vision(*_args: Any) -> None:
                self.torch.cuda.synchronize()
                vision_started[0] = perf_counter()

            def after_vision(*_args: Any) -> None:
                self.torch.cuda.synchronize()
                vision_seconds[0] += perf_counter() - vision_started[0]

            pre_hook = self.model.visual.register_forward_pre_hook(before_vision)
            post_hook = self.model.visual.register_forward_hook(after_vision)
            generation_started = perf_counter()
            try:
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        generation_config=self.generation_config,
                        use_model_defaults=False,
                        **generation_kwargs,
                    )
            finally:
                pre_hook.remove()
                post_hook.remove()
            generation_seconds = perf_counter() - generation_started
            prompt_length = prepared.prompt_length
        prediction_ids = generated[:, prompt_length:] if generated.shape[-1] > prompt_length else generated
        with self._processor_lock:
            texts = self.processor.batch_decode(
                prediction_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
        stats_batch = list(getattr(self.model, "fastvid_last_stats_batch", []) or [])
        aggregate_stats = dict(getattr(self.model, "fastvid_last_stats", {}) or {})
        if not stats_batch and len(prepared.requests) == 1:
            stats_batch = [aggregate_stats]
        allocator_stats_all = list(getattr(self.model, "last_tpsa_stats_batch", []) or [])
        allocator_stats_batch = (
            allocator_stats_all[-len(prepared.requests):] if allocator_stats_all else []
        )
        batch_wall = perf_counter() - started
        valid_tokens = inputs.get("attention_mask")
        first_token_topk: list[dict[str, list[float] | list[int]]] = []
        if logits_capture is not None and logits_capture.values is not None:
            count = min(8, int(logits_capture.values.shape[-1]))
            top_values, top_indices = self.torch.topk(logits_capture.values, k=count, dim=-1)
            first_token_topk = [
                {
                    "token_ids": [int(value) for value in top_indices[row].tolist()],
                    "logits": [float(value) for value in top_values[row].tolist()],
                }
                for row in range(top_indices.shape[0])
            ]
        outputs: list[GroundingPrediction | Exception] = []
        for batch_index, (request, text) in enumerate(zip(prepared.requests, texts)):
            semvid_stats = dict(stats_batch[batch_index]) if batch_index < len(stats_batch) else {}
            allocator_stats = (
                dict(allocator_stats_batch[batch_index])
                if batch_index < len(allocator_stats_batch) else {}
            )
            semvid_stats.update(allocator_stats)
            grid = inputs.get("video_grid_thw")
            if (
                self.observation.policy == "single_pass"
                and self.torch.is_tensor(grid)
                and batch_index < grid.shape[0]
            ):
                merge = int(getattr(self.model.visual, "spatial_merge_size", 1))
                dense_visual_tokens = int(grid[batch_index].prod().item()) // (merge * merge)
                grid_frames = int(grid[batch_index, 0].item())
                grid_patches = dense_visual_tokens // max(grid_frames, 1)
            else:
                dense_visual_tokens = 0
                grid_frames = grid_patches = 0
            if dense_visual_tokens and not semvid_stats.get("orig_video_tokens"):
                semvid_stats["orig_video_tokens"] = dense_visual_tokens
            if (
                dense_visual_tokens
                and self.spatial_allocator.spatial_policy == "dense"
                and not semvid_stats.get("kept_video_tokens")
            ):
                semvid_stats["kept_video_tokens"] = dense_visual_tokens
            row_valid_tokens = (
                int(valid_tokens[batch_index].sum().item())
                if self.torch.is_tensor(valid_tokens) else prompt_length
            )
            role_stats = semvid_stats if self.observation.policy == "hmve" else allocator_stats
            token_roles = _token_role_counts(self.model, batch_index, role_stats)
            per_frame_allocation = _per_frame_allocation(
                self.model,
                batch_index,
                grid_frames,
                grid_patches,
                role_stats,
                self.spatial_allocator.spatial_policy,
            )
            original_prefill_length = int(semvid_stats.get("orig_seq_len", row_valid_tokens))
            compact_prefill_length = int(semvid_stats.get("new_seq_len", row_valid_tokens))
            telemetry = {
                **prepared.input_stats[batch_index],
                "input_preparation_seconds": prepared.preparation_seconds / len(prepared.requests),
                "batch_input_preparation_seconds": prepared.preparation_seconds,
                "queue_wait_seconds": max(0.0, started - prepared.ready_at) / len(prepared.requests),
                "batch_queue_wait_seconds": max(0.0, started - prepared.ready_at),
                "host_to_device_seconds": transfer_seconds / len(prepared.requests),
                "batch_host_to_device_seconds": transfer_seconds,
                "vision_encoder_seconds": vision_seconds[0] / len(prepared.requests),
                "generation_seconds": generation_seconds / len(prepared.requests),
                "batch_generation_seconds": generation_seconds,
                "component_seconds": batch_wall / len(prepared.requests),
                "qwen_batch_size": len(prepared.requests),
                "model_gpu_memory_ratio": self.config.model_gpu_memory_ratio,
                "batch_padding_tokens": prompt_length - row_valid_tokens,
                "pinned_memory_bytes": prepared.pinned_memory_bytes,
                "prefill_tokens_before_pruning": original_prefill_length,
                "prefill_tokens_after_pruning": compact_prefill_length,
                "original_prefill_length": original_prefill_length,
                "compact_prefill_length": compact_prefill_length,
                "target_retained_tokens": int(semvid_stats.get(
                    "target_retained_tokens", semvid_stats.get("kept_video_tokens", 0)
                )),
                "actual_retained_tokens": int(semvid_stats.get(
                    "actual_retained_tokens", semvid_stats.get("kept_video_tokens", 0)
                )),
                "effective_retention_ratio": float(semvid_stats.get(
                    "effective_retention_ratio",
                    float(semvid_stats.get("kept_video_tokens", 0)) /
                    max(int(semvid_stats.get("orig_video_tokens", 0)), 1),
                )),
                "per_frame_allocation": per_frame_allocation,
                "selected_boundary_bands": {
                    "start": semvid_stats.get("start_boundary_bands", []),
                    "end": semvid_stats.get("end_boundary_bands", []),
                },
                "query_allocation_seconds": float(semvid_stats.get("query_allocation_seconds", 0.0)),
                "motion_allocation_seconds": float(semvid_stats.get("motion_allocation_seconds", 0.0)),
                "boundary_allocation_seconds": float(semvid_stats.get("boundary_allocation_seconds", 0.0)),
                "selection_seconds": float(semvid_stats.get("selection_seconds", 0.0)),
                "query_only_overlap_tokens": int(
                    semvid_stats.get("query_only_overlap_tokens", 0)
                ),
                "query_only_overlap_fraction": float(
                    semvid_stats.get("query_only_overlap_fraction", 0.0)
                ),
                "query_only_nonprototype_overlap_tokens": int(
                    semvid_stats.get("query_only_nonprototype_overlap_tokens", 0)
                ),
                "query_only_nonprototype_overlap_fraction": float(
                    semvid_stats.get("query_only_nonprototype_overlap_fraction", 0.0)
                ),
                "attempted_replacements": int(semvid_stats.get("attempted_replacements", 0)),
                "actual_replacements": int(semvid_stats.get("actual_replacements", 0)),
                "attempted_motion_replacements": int(
                    semvid_stats.get("attempted_motion_replacements", 0)
                ),
                "actual_motion_replacements": int(
                    semvid_stats.get("actual_motion_replacements", 0)
                ),
                "attempted_boundary_replacements": int(
                    semvid_stats.get("attempted_boundary_replacements", 0)
                ),
                "actual_boundary_replacements": int(
                    semvid_stats.get("actual_boundary_replacements", 0)
                ),
                "motion_gated_frames": semvid_stats.get("motion_gated_frames", []),
                "motion_gated_frame_count": int(
                    semvid_stats.get("motion_gated_frame_count", 0)
                ),
                "rejected_boundary_bands": {
                    "start": int(semvid_stats.get("rejected_start_boundary_bands", 0)),
                    "end": int(semvid_stats.get("rejected_end_boundary_bands", 0)),
                },
                "boundary_evidence": {
                    "start": semvid_stats.get("start_evidence", {}),
                    "end": semvid_stats.get("end_evidence", {}),
                },
                "auxiliary_quota": int(semvid_stats.get("auxiliary_quota", 0)),
                "quota_returned_to_query": int(
                    semvid_stats.get("quota_returned_to_query", 0)
                ),
                "decoded_frames": int(semvid_stats.get(
                    "decoded_frames", prepared.input_stats[batch_index].get("decoded_frames", 0),
                )),
                "decoded_pixels": int(semvid_stats.get(
                    "decoded_pixels", prepared.input_stats[batch_index].get("decoded_pixels", 0),
                )),
                "hmve": semvid_stats.get("hmve"),
                "generated_tokens": int(prediction_ids.shape[-1]),
                "first_token_topk": (
                    first_token_topk[batch_index] if batch_index < len(first_token_topk) else None
                ),
            }
            parse_status = "parsed"
            try:
                if request.sample.cardinality == "multi":
                    parsed = parse_intervals_detailed(text)
                    response_intervals = parsed.intervals
                    parse_status = parsed.status
                else:
                    response_interval = parse_timestamp(text)
                    response_intervals = (response_interval,)
            except (TypeError, ValueError) as error:
                outputs.append(GroundingOutputError(
                    str(error), text, semvid_stats, token_roles, telemetry,
                ))
                continue
            try:
                intervals = tuple(
                    normalize_timestamp(
                        candidate, request.component, request.sample.duration,
                        "absolute",
                    )
                    for candidate in response_intervals
                )
                if request.sample.cardinality == "multi":
                    intervals = consolidate_intervals(
                        intervals,
                        duration=request.sample.duration,
                    )
                interval = intervals[0] if intervals else None
            except (TypeError, ValueError) as error:
                outputs.append(GroundingOutputError(
                    str(error), text, semvid_stats, token_roles, telemetry,
                ))
                continue
            outputs.append(GroundingPrediction(
                interval=interval, component=request.component, raw_text=text,
                spatial_stats=semvid_stats,
                token_roles=token_roles,
                telemetry=telemetry,
                intervals=intervals,
                parse_status=parse_status,
            ))
        return outputs

    def ground_batch(
        self, requests: Sequence[GroundingRequest], *, pin_memory: bool = False,
    ) -> list[GroundingPrediction | Exception]:
        return self.ground_prepared(self.prepare_batch(requests, pin_memory=pin_memory))

    def ground(self, sample: Sample, component: Component) -> GroundingPrediction:
        result = self.ground_batch([GroundingRequest(sample, component)])[0]
        if isinstance(result, Exception):
            raise result
        return result


# Neutral name for new integrations; old imports remain valid.
QwenGrounder = SemVIDGrounder
