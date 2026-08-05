"""Focused adapter from routed clips to the official SemVID Qwen implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

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


def _token_role_counts(model: Any) -> dict[str, int]:
    coordinates = getattr(model, "last_semantic_prune_coords", None) or {}
    output = {}
    for role in ("context", "object", "motion"):
        values = coordinates.get(role, [])
        output[role] = sum(int(value.numel() // 2) for value in values if hasattr(value, "numel"))
    return output


def _render_generation_prompt(processor: Any, prompt: list[dict[str, Any]], force_stop_thinking: bool) -> str:
    """Render a Qwen prompt using the same thinking control as SemVID evaluation."""
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if force_stop_thinking:
        text += "</think>"
    return text


class SemVIDGrounder:
    """One-model, batch-size-one inference adapter for routed video components."""

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
        self.model.eval().requires_grad_(False)
        self.config = config
        self.torch = torch
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

    def _prepare(self, prompt: list[dict[str, Any]], query: str) -> tuple[dict[str, Any], int, dict[str, int]]:
        try:
            import qwen_vl_utils
        except ImportError as error:
            raise RuntimeError("qwen-vl-utils is required by SemVID") from error
        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
            [prompt], image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
        )
        metadata = None
        if video_inputs is not None:
            video_inputs, metadata = zip(*video_inputs)
            video_inputs, metadata = list(video_inputs), list(metadata)
        decoded_frames = 0
        decoded_pixels = 0
        for video in video_inputs or []:
            shape = tuple(int(value) for value in getattr(video, "shape", ()))
            if len(shape) >= 4:
                decoded_frames += shape[0]
                decoded_pixels += shape[0] * shape[-2] * shape[-1]
        text = _render_generation_prompt(self.processor, prompt, self.config.force_stop_thinking)
        processor_kwargs = {"do_sample_frames": video_kwargs.get("do_sample_frames", False)}
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, video_metadata=metadata,
            return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False,
            do_resize=False, **processor_kwargs,
        )
        query_tokens = self.processor.tokenizer(
            [query], padding=True, truncation=True, return_tensors="pt", add_special_tokens=True,
        )
        device = getattr(self.model, "device", self.torch.device("cuda:0"))
        inputs = {key: value.to(device) if self.torch.is_tensor(value) else value for key, value in inputs.items()}
        query_tokens = {
            key: value.to(device) if self.torch.is_tensor(value) else value for key, value in query_tokens.items()
        }
        inputs["query_ids"] = query_tokens["input_ids"]
        inputs["query_attention_mask"] = query_tokens.get("attention_mask")
        return inputs, int(inputs["input_ids"].shape[-1]), {
            "decoded_frames": decoded_frames,
            "decoded_pixels": decoded_pixels,
        }

    def ground(self, sample: Sample, component: Component) -> GroundingPrediction:
        started = perf_counter()
        inputs, prompt_length, input_stats = self._prepare(self._prompt(sample, component), sample.query)
        prepared_at = perf_counter()
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
                    **inputs, generation_config=self.generation_config, use_model_defaults=False,
                )
        finally:
            pre_hook.remove()
            post_hook.remove()
        generation_seconds = perf_counter() - generation_started
        prediction_ids = generated[:, prompt_length:] if generated.shape[-1] > prompt_length else generated
        text = self.processor.batch_decode(
            prediction_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        interval = normalize_timestamp(
            parse_timestamp(text), component, sample.duration, self.config.timestamp_mode,
        )
        semvid_stats = dict(getattr(self.model, "fastvid_last_stats", {}) or {})
        return GroundingPrediction(
            interval=interval, component=component, raw_text=text,
            semvid_stats=semvid_stats,
            token_roles=_token_role_counts(self.model),
            telemetry={
                **input_stats,
                "input_preparation_seconds": prepared_at - started,
                "vision_encoder_seconds": vision_seconds[0],
                "generation_seconds": generation_seconds,
                "component_seconds": perf_counter() - started,
                "prefill_tokens_before_pruning": prompt_length,
                "prefill_tokens_after_pruning": int(semvid_stats.get("new_seq_len", prompt_length)),
                "generated_tokens": int(prediction_ids.shape[-1]),
            },
        )
