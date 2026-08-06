"""Focused adapter from routed clips to the official SemVID Qwen implementation."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .config import SemVIDConfig
from .timestamps import normalize_timestamp, parse_timestamp
from .types import Component, GroundingPrediction, Sample


HANDLER = "modeling_qwen3_vl_semvid.Qwen3VLForConditionalGenerationSemVID"


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


def _token_role_counts(model: Any, batch_index: int = 0) -> dict[str, int]:
    coordinates = getattr(model, "last_semantic_prune_coords", None) or {}
    output = {}
    for role in ("context", "object", "motion"):
        values = coordinates.get(role, [])
        value = values[batch_index] if batch_index < len(values) else None
        output[role] = int(value.numel() // 2) if hasattr(value, "numel") else 0
    return output


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

    def __init__(self, config: SemVIDConfig, semvid_root: Path | None = None) -> None:
        try:
            import torch
            from transformers import GenerationConfig
        except ImportError as error:
            raise RuntimeError("install SemVID/requirements.txt before loading the grounder") from error
        if not torch.cuda.is_available():
            raise RuntimeError("SemVID grounding requires a CUDA GPU; CPU-only routing remains available")
        root = semvid_root or default_semvid_root()
        _activate_upstream(root)
        model_source = root / "src/models/models/modeling_qwen3_vl_semvid.py"
        if config.batch_size > 1 and "pruned_attention_mask" not in model_source.read_text(encoding="utf-8"):
            raise RuntimeError(
                "Qwen microbatching requires the bundled SemVID cache/position patch. "
                "Run `bash hybrid_vtg/scripts/apply_semvid_patches.sh` first."
            )
        from src.utils.model_utils import load_hosted_model

        hyperparameters = {
            "enable_semantic_prune": config.enabled,
            "semantic_retention_ratio": config.retention_ratio,
            "semantic_stage1_topk_segments": 0,
            "semantic_stage1_smooth_win": 1,
            "semantic_frame_weight_alpha": config.frame_weight_alpha,
            "semantic_obj_ratio": config.object_ratio,
            "semantic_mmr_lambda": config.mmr_lambda,
            "semantic_min_tokens_per_frame": 0,
            "semantic_motion_query_beta": config.motion_query_beta,
            "ablation_uniform_allocation": False,
            "ablation_use_semantic_selection": "semvid",
            "attn_implementation": config.attention,
        }
        self.model, self.processor = load_hosted_model(
            config.model, model_handler=HANDLER, model_hyper_parameters=hyperparameters,
            dtype=config.dtype, backend="vllm",
        )
        if self.model.config.text_config.pad_token_id is None:
            self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.eval().requires_grad_(False)
        self.config = config
        self.torch = torch
        self._processor_lock = threading.Lock()
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens, do_sample=False,
        )

    def _prompt(self, sample: Sample, component: Component) -> list[dict[str, Any]]:
        total_pixels = self.config.total_pixel_tokens * 16 * 16 * 4
        minimum_pixels = self.config.minimum_pixel_tokens * 16 * 16 * 4
        instruction = (
            f"The visible clip is from {component.start:.3f} to {component.end:.3f} seconds of the original video. "
            f"Query: {sample.query}\n"
            "When does the described event occur? Return original-video timestamps, not clip-relative time. "
            f"The answer must lie inside [{component.start:.3f}, {component.end:.3f}]. "
            'Return only JSON: {"start": number, "end": number}.'
        )
        return [{
            "role": "user",
            "content": [{
                "type": "video", "video": sample.video_path,
                "video_start": component.start, "video_end": component.end,
                "fps": self.config.fps,
                "total_pixels": total_pixels, "min_pixels": minimum_pixels,
                "max_frames": self.config.max_frames,
            }, {"type": "text", "text": instruction}],
        }]

    @staticmethod
    def _video_stats(video: Any) -> dict[str, int]:
        shape = tuple(int(value) for value in getattr(video, "shape", ()))
        if len(shape) < 4:
            return {"decoded_frames": 0, "decoded_pixels": 0}
        return {
            "decoded_frames": shape[0],
            "decoded_pixels": shape[0] * shape[-2] * shape[-1],
        }

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
        logits_capture = _FirstLogitsCapture() if self.config.capture_validation_logits else None
        generation_kwargs = {"logits_processor": [logits_capture]} if logits_capture is not None else {}
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs, generation_config=self.generation_config, use_model_defaults=False,
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
            try:
                interval = normalize_timestamp(
                    parse_timestamp(text), request.component, request.sample.duration, self.config.timestamp_mode,
                )
            except (TypeError, ValueError) as error:
                outputs.append(error)
                continue
            row_valid_tokens = (
                int(valid_tokens[batch_index].sum().item()) if self.torch.is_tensor(valid_tokens) else prompt_length
            )
            outputs.append(GroundingPrediction(
                interval=interval, component=request.component, raw_text=text,
                semvid_stats=semvid_stats,
                token_roles=_token_role_counts(self.model, batch_index),
                telemetry={
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
                    "batch_padding_tokens": prompt_length - row_valid_tokens,
                    "pinned_memory_bytes": prepared.pinned_memory_bytes,
                    "prefill_tokens_before_pruning": row_valid_tokens,
                    "prefill_tokens_after_pruning": int(semvid_stats.get("new_seq_len", row_valid_tokens)),
                    "generated_tokens": int(prediction_ids.shape[-1]),
                    "first_token_topk": (
                        first_token_topk[batch_index] if batch_index < len(first_token_topk) else None
                    ),
                },
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
