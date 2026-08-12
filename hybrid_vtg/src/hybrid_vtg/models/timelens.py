"""TimeLens native controls and Qwen2.5-VL evidence integration."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

from ..contracts import Prediction, Sample
from ..postprocess import parse_spans
from .qwen import _attention_options, _generation_token_budget
from .unitime import VISION_END_TOKEN, VISION_START_TOKEN, UniTimeEvidenceBackend, compact_mrope_positions

IMAGE_TOKEN = "<|image_pad|>"
TIMELENS_VISUAL_TOKEN_BUDGET = 4_096


def require_native_video_reader() -> None:
    """Fail before loading model weights when the official video reader is absent."""
    if importlib.util.find_spec("decord") is None:
        raise RuntimeError(
            "native TimeLens inference requires decord because recent torchvision builds "
            "do not provide torchvision.io.read_video; install with: "
            "pip install 'qwen-vl-utils[decord]>=0.0.14'"
        )


def native_timelens_prediction(model: Any, processor: Any, sample: Sample, *, family: str) -> Prediction:
    """Run the released checkpoint using the corresponding official model-card recipe."""
    import torch
    from qwen_vl_utils import process_vision_info

    if family == "qwen3":
        prompt = (
            f'Given the query: "{sample.query}", return ALL time spans (in seconds) where the query is relevant.\n'
            "Output format MUST be a JSON array of [start, end] pairs.\n"
        )
        video = {
            "type": "video",
            "video": sample.video_path.resolve().as_uri(),
            "fps": 2.0,
            "min_pixels": 32 * 32,
            "max_pixels": 480 * 480,
            "total_pixels": TIMELENS_VISUAL_TOKEN_BUDGET * 32 * 32,
        }
    elif family == "qwen2.5":
        prompt = (
            "You are given a video with multiple frames. The numbers before each video frame indicate its "
            "sampling timestamp (in seconds). Please find the visual event described by the sentence "
            f"'{sample.query}', determining its starting and ending times. The format should be: "
            "'The event happens in <start time> - <end time> seconds'."
        )
        video = {
            "type": "video",
            "video": str(sample.video_path.resolve()),
            "fps": 2.0,
            "min_pixels": 64 * 28 * 28,
            "total_pixels": TIMELENS_VISUAL_TOKEN_BUDGET * 28 * 28,
        }
    else:
        raise ValueError(f"unsupported TimeLens family: {family}")

    messages = [{"role": "user", "content": [video, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if family == "qwen3":
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        inputs = processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
    else:
        images, videos = process_vision_info(messages, return_video_metadata=True)
        inputs = processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )

    device = next(model.get_input_embeddings().parameters()).device
    inputs = inputs.to(device)
    max_new_tokens = _generation_token_budget(sample)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            logits_to_keep=1,
        )
    output = generated[:, inputs.input_ids.shape[1] :]
    raw = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    spans = tuple(value for span in parse_spans(raw) if (value := span.clipped(sample.duration)))
    return Prediction(
        spans,
        raw,
        {
            "backend": "timelens-native",
            "checkpoint_family": family,
            "native_whole_video_control": True,
            "visual_token_budget": TIMELENS_VISUAL_TOKEN_BUDGET,
            "fps": 2.0,
            "max_new_tokens": max_new_tokens,
        },
    )


def _model_class():
    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration


def _install_mage_qwen25_vision_pruning(model: Any, prune_layer: int) -> None:
    """Prune complete merger cells inside Qwen2.5-VL's windowed vision encoder."""
    import torch
    import torch.nn.functional as functional

    visual = model.model.visual
    original_forward = visual.forward
    merge = int(model.config.vision_config.spatial_merge_size)
    merge_unit = merge * merge
    if not 0 <= prune_layer < len(visual.blocks):
        raise ValueError(f"encoder prune layer must be between 0 and {len(visual.blocks) - 1}")

    def sparse_forward(hidden_states, grid_thw, **kwargs):
        plan = getattr(model.model, "_mage_prune_plan", None)
        if plan is None:
            return original_forward(hidden_states, grid_thw, **kwargs)
        if int(grid_thw.shape[0]) != 1:
            raise ValueError("Mage-style TimeLens-7B pruning currently supports batch size one")

        hidden_states = visual.patch_embed(hidden_states)
        rotary = visual.rot_pos_emb(grid_thw)
        window_index, window_boundaries = visual.get_window_index(grid_thw)
        window_boundaries = torch.tensor(window_boundaries, device=hidden_states.device, dtype=torch.int32)
        window_boundaries = torch.unique_consecutive(window_boundaries)
        sequence = int(hidden_states.shape[0])
        hidden_states = hidden_states.reshape(sequence // merge_unit, merge_unit, -1)[window_index]
        hidden_states = hidden_states.reshape(sequence, -1)
        rotary = rotary.reshape(sequence // merge_unit, merge_unit, -1)[window_index].reshape(sequence, -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        full_boundaries = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        full_boundaries = functional.pad(full_boundaries, (1, 0), value=0)
        selected_original = None

        for layer_num, block in enumerate(visual.blocks):
            if layer_num == prune_layer:
                selected_cells = torch.tensor(
                    [value // merge_unit for value in plan.patch_indices[::merge_unit]],
                    device=window_index.device,
                    dtype=window_index.dtype,
                )
                keep_cell = torch.isin(window_index, selected_cells)
                kept_window_cells = torch.nonzero(keep_cell, as_tuple=False).flatten()
                patch_offsets = torch.arange(merge_unit, device=hidden_states.device)
                keep_patches = (kept_window_cells[:, None] * merge_unit + patch_offsets).reshape(-1)
                hidden_states = hidden_states[keep_patches]
                cos, sin = position_embeddings
                position_embeddings = (cos[keep_patches], sin[keep_patches])
                selected_original = window_index[kept_window_cells]

                window_counts = []
                for start, end in zip(window_boundaries[:-1], window_boundaries[1:]):
                    count = int(((keep_patches >= start) & (keep_patches < end)).sum())
                    if count:
                        window_counts.append(count)
                window_boundaries = functional.pad(
                    torch.tensor(window_counts, device=grid_thw.device, dtype=torch.int32).cumsum(0),
                    (1, 0),
                    value=0,
                )
                per_time = torch.tensor(
                    [count * merge_unit for count in plan.cells_per_time],
                    device=grid_thw.device,
                    dtype=torch.int32,
                )
                full_boundaries = functional.pad(per_time.cumsum(0), (1, 0), value=0)

            boundaries = full_boundaries if layer_num in visual.fullatt_block_indexes else window_boundaries
            hidden_states = block(
                hidden_states,
                cu_seqlens=boundaries,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = visual.merger(hidden_states)
        if selected_original is None:
            raise RuntimeError("Mage pruning layer was not reached")
        return hidden_states[torch.argsort(selected_original)]

    visual.forward = sparse_forward


class TimeLens7EvidenceBackend(UniTimeEvidenceBackend):
    """Frozen TimeLens-7B with native and HMVE-compatible evidence paths."""

    capabilities = frozenset(
        {"encoded-evidence", "spatial-evidence", "generative", "timestamp-interleaved", "native-video-grounding"}
    )

    def __init__(
        self,
        checkpoint: str,
        cache_dir: Path,
        *,
        encoder_pruning: str = "none",
        encoder_retention: float = 1.0,
        encoder_prune_layer: int = 0,
        post_pruning: str = "none",
        post_retention: float = 1.0,
    ) -> None:
        super().__init__(
            None,
            cache_dir,
            base_checkpoint=checkpoint,
            name="timelens-7b",
            encoder_pruning=encoder_pruning,
            encoder_retention=encoder_retention,
            encoder_prune_layer=encoder_prune_layer,
            post_pruning=post_pruning,
            post_retention=post_retention,
        )
        self._native_processor: Any = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            from transformers import AutoProcessor, Qwen2_5_VLProcessor
            from transformers import logging as hf_logging

            previous = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                base = _model_class().from_pretrained(
                    self.base_checkpoint,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    **_attention_options(),
                ).eval()
                self._model = self._base_model = base
                # The checkpoint's remote TimeLensProcessor accepts only raw
                # video tensors paired with metadata. Our adaptive path passes
                # explicitly sampled PIL frames, so it must use the standard
                # Qwen2.5-VL processor from the same checkpoint instead.
                self._processor = Qwen2_5_VLProcessor.from_pretrained(
                    self.base_checkpoint,
                    padding_side="left",
                    do_resize=False,
                    use_fast=False,
                )
                self._native_processor = AutoProcessor.from_pretrained(
                    self.base_checkpoint,
                    padding_side="left",
                    do_resize=False,
                    trust_remote_code=True,
                    use_fast=False,
                )
                if self.encoder_pruning == "mage":
                    _install_mage_qwen25_vision_pruning(base, self.encoder_prune_layer)
            finally:
                hf_logging.set_verbosity(previous)
        return self._model, self._base_model, self._processor

    @staticmethod
    def _instruction(sample: Sample, *, coarse: bool = False) -> str:
        del coarse
        if sample.cardinality == "single":
            return (
                f"Please find the visual event described by the sentence '{sample.query}', determining its "
                "starting and ending times. The format should be: "
                "'The event happens in <start time> - <end time> seconds'."
            )
        return (
            f"Find every separate temporal window where '{sample.query}' occurs. Return ONLY a strict JSON array "
            "of [start, end] pairs using the displayed timestamp seconds, or [] if it is absent. Do not merge "
            "nearby but distinct occurrences."
        )

    def _evidence_prompt(self, sample: Sample, evidence, *, segment_seconds: float | None = None):
        """Build the per-frame image-token layout used to train TimeLens-7B."""
        import torch

        if segment_seconds is not None:
            raise ValueError("TimeLens-7B does not support UniTime's trained coarse retrieval prompt")
        _, base, processor = self._load()
        coordinates = evidence.metadata.get("cell_coordinates")
        valid_coordinates = isinstance(coordinates, list) and len(coordinates) == evidence.size
        groups: list[tuple[float, list[int]]] = []
        for index, timestamp in enumerate(evidence.timestamps):
            if groups and abs(groups[-1][0] - timestamp) < 1e-6:
                groups[-1][1].append(index)
            else:
                groups.append((timestamp, [index]))

        coordinate_groups = []
        visual_parts = []
        for timestamp, indices in groups:
            fallback_side = max(1, math.ceil(math.sqrt(len(indices))))
            local = (
                [[0, *coordinates[index][-2:]] for index in indices]
                if valid_coordinates
                else [[0, index // fallback_side, index % fallback_side] for index in range(len(indices))]
            )
            coordinate_groups.append(local)
            visual_parts.append(
                f"{timestamp:.1f}s: " + VISION_START_TOKEN + IMAGE_TOKEN * len(indices) + VISION_END_TOKEN
            )
        prompt = (
            f"<|im_start|>user\n{''.join(visual_parts)}{self._instruction(sample)}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        encoded = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        text_embeddings = base.get_input_embeddings()
        device = self._device(text_embeddings)
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        slots = torch.nonzero(input_ids[0] == base.config.image_token_id, as_tuple=False).flatten()
        if slots.numel() != evidence.size:
            raise RuntimeError(f"TimeLens prompt has {slots.numel()} image slots for {evidence.size} evidence rows")
        with torch.inference_mode():
            inputs_embeds = text_embeddings(input_ids)
            inputs_embeds[0, slots] = evidence.embeddings.to(device, inputs_embeds.dtype)
        positions = compact_mrope_positions(input_ids, base.config.image_token_id, coordinate_groups)
        base.model.rope_deltas = positions.max().reshape(1, 1) + 1 - input_ids.shape[1]
        return input_ids, attention, inputs_embeds, positions, [timestamp for timestamp, _ in groups]

    def predict_video(self, sample: Sample) -> Prediction:
        require_native_video_reader()
        model, _, _ = self._load()
        return native_timelens_prediction(model, self._native_processor, sample, family="qwen2.5")


__all__ = [
    "TIMELENS_VISUAL_TOKEN_BUDGET",
    "TimeLens7EvidenceBackend",
    "_install_mage_qwen25_vision_pruning",
    "native_timelens_prediction",
    "require_native_video_reader",
]
