"""Clean-room frozen UniTime adapter backend built on standard Qwen2-VL APIs."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from ..contracts import GroundingContext, ModelBackend, Prediction, Sample, ScoredSpan, TemporalEvidence
from ..media import extract_frames
from ..postprocess import consolidate_spans, parse_spans
from .pruning import mage_cell_plan, motion_residual_importance, semvid_select
from .qwen import _dense_evidence_units, _generation_token_budget

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
VIDEO_TOKEN = "<|video_pad|>"
UNITIME_EVIDENCE_BUDGET = 4_096


def adaptive_frame_size(width: int, height: int, target_cells: int) -> tuple[int, int]:
    """Return a 28-aligned frame size no larger than a merger-cell budget."""
    if width <= 0 or height <= 0 or target_cells <= 0:
        raise ValueError("frame dimensions and target cells must be positive")
    cell_height = max(1, round(math.sqrt(target_cells * height / width)))
    cell_width = max(1, target_cells // cell_height)
    while cell_height * cell_width > target_cells:
        if cell_width >= cell_height:
            cell_width -= 1
        else:
            cell_height -= 1
    return cell_width * 28, cell_height * 28


class _CompactQwen2Mixin:
    """Keep sparse MRoPE and avoid full-vocabulary logits for every prefill token."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        logits_to_keep: int = 0,
        **kwargs: Any,
    ):
        """Backport Qwen3's selective-logits inference path to Qwen2-VL.

        Transformers 4.57 applies Qwen2-VL's language head to the complete
        prefill. At 16,384 tokens that allocates about 5.09 GB solely for BF16
        logits. Generation needs only the final position, so temporarily slice
        the hidden states entering the frozen head when requested.
        """
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be non-negative")
        if labels is not None and logits_to_keep:
            raise ValueError("selective logits are available only for frozen inference")
        forwarded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "rope_deltas": rope_deltas,
            "cache_position": cache_position,
            **kwargs,
        }
        if not logits_to_keep:
            return super().forward(**forwarded)

        original_forward = self.lm_head.forward

        def compact_head(hidden_states):
            return original_forward(hidden_states[:, -logits_to_keep:, :])

        self.lm_head.forward = compact_head
        try:
            return super().forward(**forwarded)
        finally:
            self.lm_head.forward = original_forward

    def prepare_inputs_for_generation(
        self,
        *args: Any,
        position_ids=None,
        cache_position=None,
        inputs_embeds=None,
        **kwargs: Any,
    ):
        prefill = cache_position is None or int(cache_position[0]) == 0
        # The explicit sparse grid is valid only for the prefill. On cached
        # decoding steps, let Qwen2 derive the next position from rope_deltas;
        # reusing the full prefill tensor broadcasts an extra sequence axis.
        forwarded_positions = position_ids if prefill else None
        values = super().prepare_inputs_for_generation(
            *args,
            position_ids=forwarded_positions,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if position_ids is not None and prefill:
            values["position_ids"] = position_ids
        return values


def _model_class():
    from transformers import Qwen2VLForConditionalGeneration

    class CompactQwen2(_CompactQwen2Mixin, Qwen2VLForConditionalGeneration):
        pass

    return CompactQwen2


def _install_mage_qwen2_vision_pruning(model: Any, prune_layer: int) -> None:
    """Prune complete merger cells before one frozen Qwen2 vision block."""
    import torch
    import torch.nn.functional as functional

    visual = model.model.visual
    original_forward = visual.forward
    merge = int(model.config.vision_config.spatial_merge_size)
    if not 0 <= prune_layer < len(visual.blocks):
        raise ValueError(f"encoder prune layer must be between 0 and {len(visual.blocks) - 1}")

    def sparse_forward(hidden_states, grid_thw, **kwargs):
        plan = getattr(model.model, "_mage_prune_plan", None)
        if plan is None:
            return original_forward(hidden_states, grid_thw, **kwargs)
        if int(grid_thw.shape[0]) != 1:
            raise ValueError("Mage-style UniTime pruning currently supports batch size one")

        hidden_states = visual.patch_embed(hidden_states)
        rotary = visual.rot_pos_emb(grid_thw)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = functional.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, block in enumerate(visual.blocks):
            if layer_num == prune_layer:
                index = torch.tensor(plan.patch_indices, device=hidden_states.device, dtype=torch.long)
                hidden_states = hidden_states[index]
                cos, sin = position_embeddings
                position_embeddings = (cos[index], sin[index])
                per_time = torch.tensor(
                    [count * merge * merge for count in plan.cells_per_time],
                    device=grid_thw.device,
                    dtype=torch.int32,
                )
                cu_seqlens = functional.pad(per_time.cumsum(0, dtype=torch.int32), (1, 0), value=0)
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return visual.merger(hidden_states)

    visual.forward = sparse_forward


def compact_mrope_positions(input_ids, video_token_id: int, coordinate_groups: Sequence[Sequence[Sequence[int]]]):
    """Construct Qwen2 MRoPE positions for variable sparse visual-token runs."""
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("compact UniTime prompting supports batch size one")
    positions = torch.zeros((3, 1, input_ids.shape[1]), device=input_ids.device, dtype=torch.long)
    tokens = input_ids[0]
    cursor = 0
    prior_max = -1
    group_index = 0
    while cursor < tokens.numel():
        matches = torch.nonzero(tokens[cursor:] == video_token_id, as_tuple=False).flatten()
        if not matches.numel():
            count = tokens.numel() - cursor
            start = prior_max + 1
            positions[:, 0, cursor:] = torch.arange(start, start + count, device=tokens.device)
            cursor = tokens.numel()
            break

        visual_start = cursor + int(matches[0])
        text_count = visual_start - cursor
        text_start = prior_max + 1
        if text_count:
            positions[:, 0, cursor:visual_start] = torch.arange(
                text_start,
                text_start + text_count,
                device=tokens.device,
            )
        base = text_start + text_count
        visual_end = visual_start
        while visual_end < tokens.numel() and int(tokens[visual_end]) == video_token_id:
            visual_end += 1
        count = visual_end - visual_start
        if group_index >= len(coordinate_groups):
            raise ValueError("prompt contains more visual groups than evidence")
        coordinates = coordinate_groups[group_index]
        if len(coordinates) != count:
            raise ValueError(f"visual group has {count} slots but {len(coordinates)} coordinates")
        side = max(1, math.ceil(math.sqrt(count)))
        temporal_origin = min((int(value[-3]) for value in coordinates if len(value) >= 3), default=0)
        for local, coordinate in enumerate(coordinates):
            temporal = int(coordinate[-3]) - temporal_origin if len(coordinate) >= 3 else 0
            row = int(coordinate[-2]) if len(coordinate) >= 2 else local // side
            column = int(coordinate[-1]) if len(coordinate) >= 1 else local % side
            positions[0, 0, visual_start + local] = base + max(0, temporal)
            positions[1, 0, visual_start + local] = base + max(0, row)
            positions[2, 0, visual_start + local] = base + max(0, column)
        prior_max = int(positions[:, 0, cursor:visual_end].max())
        cursor = visual_end
        group_index += 1

    if group_index != len(coordinate_groups):
        raise ValueError("evidence contains more visual groups than the prompt")
    return positions


class UniTimeEvidenceBackend(ModelBackend):
    """Frozen Qwen2-VL with optional UniTime LoRA, routing, and pruning."""

    capabilities = frozenset(
        {"encoded-evidence", "spatial-evidence", "generative", "timestamp-interleaved", "unitime-coarse"}
    )

    def __init__(
        self,
        adapter_checkpoint: str | None,
        cache_dir: Path,
        *,
        base_checkpoint: str = "Qwen/Qwen2-VL-7B-Instruct",
        name: str = "unitime",
        encoder_pruning: str = "none",
        encoder_retention: float = 1.0,
        encoder_prune_layer: int = 0,
        post_pruning: str = "none",
        post_retention: float = 1.0,
    ) -> None:
        if encoder_pruning not in {"none", "mage"}:
            raise ValueError("encoder pruning must be 'none' or 'mage'")
        if post_pruning not in {"none", "semvid"}:
            raise ValueError("post pruning must be 'none' or 'semvid'")
        if not 0 < encoder_retention <= 1 or not 0 < post_retention <= 1:
            raise ValueError("pruning retention ratios must be in (0, 1]")
        if encoder_pruning == "none" and encoder_retention != 1.0:
            raise ValueError("encoder retention requires Mage pruning")
        if post_pruning == "none" and post_retention != 1.0:
            raise ValueError("post retention requires SemVID pruning")
        if encoder_pruning == "mage" and post_pruning == "semvid" and post_retention > encoder_retention:
            raise ValueError("post retention cannot exceed encoder retention")
        if encoder_prune_layer < 0:
            raise ValueError("encoder prune layer must be non-negative")
        self.name = name
        self.adapter_checkpoint = adapter_checkpoint
        self.base_checkpoint = base_checkpoint
        self.cache_dir = cache_dir
        self.encoder_pruning = encoder_pruning
        self.encoder_retention = encoder_retention
        self.encoder_prune_layer = encoder_prune_layer
        self.post_pruning = post_pruning
        self.post_retention = post_retention
        self._model: Any = None
        self._base_model: Any = None
        self._processor: Any = None

    @property
    def maximum_evidence_units(self) -> int:
        return UNITIME_EVIDENCE_BUDGET

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            from transformers import AutoProcessor
            from transformers import logging as hf_logging

            previous = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                base = _model_class().from_pretrained(
                    self.base_checkpoint,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                ).eval()
                if self.adapter_checkpoint:
                    from peft import PeftModel

                    self._model = PeftModel.from_pretrained(
                        base,
                        self.adapter_checkpoint,
                        is_trainable=False,
                    ).eval()
                else:
                    self._model = base
                self._base_model = base
                self._processor = AutoProcessor.from_pretrained(self.base_checkpoint, use_fast=False)
                if self.encoder_pruning == "mage":
                    _install_mage_qwen2_vision_pruning(base, self.encoder_prune_layer)
            finally:
                hf_logging.set_verbosity(previous)
        return self._model, self._base_model, self._processor

    @staticmethod
    def _device(module: Any):
        return next(module.parameters()).device

    def _cache_path(self, sample: Sample, timestamps: Sequence[float]) -> Path:
        stat = sample.video_path.stat()
        identity = "|".join(
            [
                str(sample.video_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                self.base_checkpoint,
                self.adapter_checkpoint or "no-adapter",
                str(self.maximum_evidence_units),
                self.encoder_pruning,
                f"{self.encoder_retention:.8f}",
                str(self.encoder_prune_layer),
                ",".join(f"{value:.6f}" for value in timestamps),
            ]
        )
        return self.cache_dir / "unitime-features" / f"{hashlib.sha256(identity.encode()).hexdigest()}.pt"

    def encode(self, sample: Sample, timestamps: Sequence[float]) -> TemporalEvidence:
        import numpy as np
        import torch
        from PIL import Image

        if not timestamps:
            raise ValueError("UniTime requires at least one timestamp")
        cache_path = self._cache_path(sample, timestamps)
        model, base, processor = self._load()
        text_embeddings = base.get_input_embeddings()
        text_device = self._device(text_embeddings)
        if cache_path.is_file():
            cache_started = perf_counter()
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            metadata = dict(payload["metadata"])
            metadata["feature_cache_hit"] = True
            metadata["timing"] = {"feature_cache_load_seconds": perf_counter() - cache_started}
            return TemporalEvidence(
                payload["embeddings"].to(text_device),
                tuple(float(value) for value in payload["timestamps"]),
                int(payload["source_frames"]),
                metadata,
            )

        decode_started = perf_counter()
        # Preserve enough source resolution for the adaptive Qwen grid. The
        # shared decoder's 336px default is suitable for the smaller backends,
        # but would force UniTime to upscale an already downsampled JPEG.
        paths = extract_frames(
            sample.video_path,
            timestamps,
            self.cache_dir / "frames",
            maximum_width=2_048,
        )
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        decode_seconds = perf_counter() - decode_started
        placeholder = VISION_START_TOKEN + VIDEO_TOKEN + VISION_END_TOKEN
        # Apply the shared 4,096-cell Qwen2 budget. Qwen2 uses two-frame
        # tubelets and one post-merger cell per 28x28 input-pixel region. Modern
        # Transformers video kwargs no longer expose Qwen's max_pixels option,
        # so resize explicitly before processing and keep the processor aligned.
        estimated_temporal = max(1, math.ceil(len(frames) / 2))
        cells_per_time = max(1, min(768, self.maximum_evidence_units // estimated_temporal))
        resized_width, resized_height = adaptive_frame_size(frames[0].width, frames[0].height, cells_per_time)
        frames = [frame.resize((resized_width, resized_height), Image.Resampling.LANCZOS) for frame in frames]
        processor_started = perf_counter()
        inputs = processor(
            text=[placeholder],
            videos=[frames],
            return_tensors="pt",
            do_resize=False,
        )
        processor_seconds = perf_counter() - processor_started
        visual = base.model.visual
        visual_device = self._device(visual)
        pixels = inputs["pixel_values_videos"].to(visual_device)
        grids = inputs["video_grid_thw"].to(visual_device)
        temporal, height, width = [int(value) for value in grids[0].tolist()]
        merge = int(base.config.vision_config.spatial_merge_size)
        dense_per_time = height * width // (merge * merge)
        plan = None
        if self.encoder_pruning == "mage":
            importance = motion_residual_importance(
                frames,
                temporal_units=temporal,
                cell_height=height // merge,
                cell_width=width // merge,
            )
            plan = mage_cell_plan(
                importance,
                merge_size=merge,
                retention_ratio=self.encoder_retention,
            )
            base.model._mage_prune_plan = plan
        try:
            vision_started = perf_counter()
            with torch.inference_mode():
                embeddings = visual(pixels.type(visual.dtype), grid_thw=grids)
            vision_seconds = perf_counter() - vision_started
        finally:
            if plan is not None:
                base.model._mage_prune_plan = None

        cells_per_time = plan.cells_per_time if plan is not None else (dense_per_time,) * temporal
        if sum(cells_per_time) != embeddings.shape[0]:
            raise RuntimeError("UniTime vision features do not match their processed grid")
        boundaries = np.linspace(0, len(timestamps), temporal + 1).round().astype(int)
        unit_times: list[float] = []
        for index in range(temporal):
            values = timestamps[boundaries[index] : max(boundaries[index] + 1, boundaries[index + 1])]
            center = sum(values) / len(values) if values else timestamps[min(index, len(timestamps) - 1)]
            unit_times.extend([float(center)] * cells_per_time[index])
        coordinates = (
            list(plan.selected_cells)
            if plan is not None
            else [
                (time, row, column)
                for time in range(temporal)
                for row in range(height // merge)
                for column in range(width // merge)
            ]
        )
        metadata = {
            "backend": self.name,
            "base_checkpoint": self.base_checkpoint,
            "adapter_checkpoint": self.adapter_checkpoint,
            "maximum_evidence_units": self.maximum_evidence_units,
            "grid_thw": [temporal, height, width],
            "tokens_per_time": list(cells_per_time),
            "cell_coordinates": [list(value) for value in coordinates],
            "dense_evidence_units": temporal * dense_per_time,
            "encoder_pruning": self.encoder_pruning,
            "encoder_retention_ratio": self.encoder_retention,
            "encoder_prune_layer": self.encoder_prune_layer,
            "encoder_retained_evidence_units": int(embeddings.shape[0]),
            "feature_cache_hit": False,
            "timing": {
                "frame_decode_seconds": decode_seconds,
                "processor_seconds": processor_seconds,
                "vision_encoder_seconds": vision_seconds,
            },
            "frame_paths": [str(path) for path in paths],
        }
        evidence = TemporalEvidence(
            embeddings.to(text_device),
            tuple(unit_times),
            len(timestamps),
            metadata,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save(
            {
                "embeddings": evidence.embeddings.detach().cpu(),
                "timestamps": list(evidence.timestamps),
                "source_frames": evidence.source_frames,
                "metadata": evidence.metadata,
            },
            temporary,
        )
        temporary.replace(cache_path)
        del model
        return evidence

    def _query_embeddings(self, query: str):
        import torch
        import torch.nn.functional as functional

        _, base, processor = self._load()
        text_embeddings = base.get_input_embeddings()
        device = self._device(text_embeddings)
        tokens = processor.tokenizer(query, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            values = text_embeddings(tokens["input_ids"].to(device))[0].float()
        return functional.normalize(values, dim=-1, eps=1e-6)

    def query_scores(self, evidence: TemporalEvidence, query: str):
        import torch.nn.functional as functional

        visual = functional.normalize(evidence.embeddings.float(), dim=-1, eps=1e-6)
        query_tokens = self._query_embeddings(query).to(visual.device)
        return (visual @ query_tokens.T).amax(-1)

    @staticmethod
    def _instruction(sample: Sample, *, coarse: bool = False) -> str:
        if coarse:
            if sample.cardinality == "multi":
                return (
                    "This sequence is interleaved with timestamps and visual features. "
                    "Identify EVERY coarse timestamp whose segment contains the query. "
                    "Return ONLY a JSON array of timestamp numbers in chronological order, "
                    "for example [0.0, 32.0, 96.0]. Return [] if it never occurs."
                )
            return (
                "This sequence is interleaved with timestamps and visual features. "
                "Identify the coarse timestamp whose segment best contains the query. "
                "Return ONLY a JSON array containing that timestamp, for example [32.0]."
            )
        if sample.cardinality == "multi":
            return (
                "This sequence is interleaved with timestamps and visual features. "
                "Identify EVERY separate temporal window where the query occurs. "
                "Keep nearby but distinct occurrences as separate pairs; do not merge them. "
                "Return ONLY a strict JSON array with one [start, end] pair per occurrence "
                "in chronological order, for example [[1.0, 3.0], [7.5, 9.0]]. "
                "Use the displayed timestamp seconds and return [] if it never occurs."
            )
        return (
            "This sequence is interleaved with timestamps and visual features. "
            "Identify the temporal window where the query occurs. Return ONLY a strict "
            "JSON array containing one [start, end] pair, or [] if it never occurs."
        )

    def _evidence_prompt(
        self,
        sample: Sample,
        evidence: TemporalEvidence,
        *,
        segment_seconds: float | None = None,
    ):
        import torch

        _, base, processor = self._load()
        coordinates = evidence.metadata.get("cell_coordinates")
        valid_coordinates = isinstance(coordinates, list) and len(coordinates) == evidence.size
        groups: list[tuple[float, int, list[int]]] = []
        for index, timestamp in enumerate(evidence.timestamps):
            key = int(timestamp // segment_seconds) if segment_seconds else len(groups)
            same_group = (
                groups
                and (
                    groups[-1][1] == key
                    if segment_seconds
                    else abs(groups[-1][0] - timestamp) < 1e-6
                )
            )
            if same_group:
                groups[-1][2].append(index)
            else:
                groups.append((timestamp, key, [index]))

        coordinate_groups = []
        visual_parts = []
        for timestamp, _, indices in groups:
            fallback_side = max(1, math.ceil(math.sqrt(len(indices))))
            local = (
                [
                    [coordinates[index][0] if segment_seconds else 0, *coordinates[index][-2:]]
                    for index in indices
                ]
                if valid_coordinates
                else [
                    [0, index // fallback_side, index % fallback_side]
                    for index in range(len(indices))
                ]
            )
            coordinate_groups.append(local)
            visual_parts.append(
                f"timestamp: {timestamp:.1f} seconds; feature: "
                + VISION_START_TOKEN
                + VIDEO_TOKEN * len(indices)
                + VISION_END_TOKEN
            )
        visual = "".join(visual_parts)
        prompt = (
            f"<|im_start|>user\n{visual}{self._instruction(sample, coarse=segment_seconds is not None)}"
            "<|im_end|>\n"
            f"<|im_start|>user\nQuery:{sample.query}\nAnswer: <|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        encoded = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        text_embeddings = base.get_input_embeddings()
        device = self._device(text_embeddings)
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        slots = torch.nonzero(input_ids[0] == base.config.video_token_id, as_tuple=False).flatten()
        if slots.numel() != evidence.size:
            raise RuntimeError(f"UniTime prompt has {slots.numel()} visual slots for {evidence.size} evidence rows")
        with torch.inference_mode():
            inputs_embeds = text_embeddings(input_ids)
            inputs_embeds[0, slots] = evidence.embeddings.to(device, inputs_embeds.dtype)
        positions = compact_mrope_positions(input_ids, base.config.video_token_id, coordinate_groups)
        base.model.rope_deltas = positions.max().reshape(1, 1) + 1 - input_ids.shape[1]
        return input_ids, attention, inputs_embeds, positions, [timestamp for timestamp, _, _ in groups]

    def _post_prune(self, sample: Sample, evidence: TemporalEvidence) -> TemporalEvidence:
        if self.post_pruning != "semvid":
            return evidence
        return semvid_select(
            evidence,
            self._query_embeddings(sample.query),
            retention_ratio=self.post_retention,
            dense_evidence_units=_dense_evidence_units(evidence.metadata, evidence.size),
        )

    def _generate(self, input_ids, attention, inputs_embeds, positions, max_new_tokens: int = 32) -> str:
        import torch

        model, _, processor = self._load()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                position_ids=positions,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                logits_to_keep=1,
            )
        output_ids = generated[:, input_ids.shape[1] :] if generated.shape[1] > input_ids.shape[1] else generated
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def coarse_corridor(
        self,
        sample: Sample,
        evidence: TemporalEvidence,
        *,
        segment_seconds: float = 32.0,
    ) -> tuple[GroundingContext, dict[str, Any]]:
        """Run UniTime's frozen fixed-segment coarse timestamp retrieval."""
        if segment_seconds <= 0:
            raise ValueError("segment length must be positive")
        pruning_started = perf_counter()
        evidence = self._post_prune(sample, evidence)
        pruning_finished = perf_counter()
        prompt_started = pruning_finished
        input_ids, attention, inputs_embeds, positions, markers = self._evidence_prompt(
            sample,
            evidence,
            segment_seconds=segment_seconds,
        )
        prompt_finished = perf_counter()
        max_new_tokens = _generation_token_budget(sample)
        raw = self._generate(input_ids, attention, inputs_embeds, positions, max_new_tokens)
        generation_finished = perf_counter()
        values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw)]
        selected = []
        for value in values:
            marker = min(markers, key=lambda timestamp: (abs(timestamp - value), timestamp))
            if marker not in selected:
                selected.append(marker)
        fallback = False
        if not selected:
            fallback = True
            scores = self.query_scores(evidence, sample.query)
            best = max(
                range(len(markers)),
                key=lambda group: float(
                    scores[
                        [
                            index
                            for index, timestamp in enumerate(evidence.timestamps)
                            if int(timestamp // segment_seconds) == int(markers[group] // segment_seconds)
                        ]
                    ].max()
                ),
            )
            selected = [markers[best]]
        start = max(0.0, min(selected))
        last = max(selected)
        following = [marker for marker in markers if marker > last]
        end = min(sample.duration, following[0] if following else last + segment_seconds)
        if end <= start:
            end = min(sample.duration, start + segment_seconds)
        return GroundingContext(start, end), {
            "coarse_raw_output": raw,
            "coarse_markers": markers,
            "coarse_selected": sorted(selected),
            "coarse_fallback": fallback,
            "coarse_evidence_units": evidence.size,
            "coarse_post_prune_seconds": pruning_finished - pruning_started,
            "coarse_prompt_seconds": prompt_finished - prompt_started,
            "coarse_generation_seconds": generation_finished - prompt_finished,
            "coarse_max_new_tokens": max_new_tokens,
        }

    def predict(self, sample: Sample, evidence: TemporalEvidence, context: GroundingContext) -> Prediction:
        del context
        pruning_started = perf_counter()
        evidence = self._post_prune(sample, evidence)
        pruning_finished = perf_counter()
        input_ids, attention, inputs_embeds, positions, _ = self._evidence_prompt(sample, evidence)
        prompt_finished = perf_counter()
        max_new_tokens = _generation_token_budget(sample)
        raw = self._generate(input_ids, attention, inputs_embeds, positions, max_new_tokens)
        generation_finished = perf_counter()
        candidates = parse_spans(raw)
        visible = sorted(set(evidence.timestamps))

        def snap(value: float) -> float:
            return min(visible, key=lambda timestamp: (abs(timestamp - value), timestamp))

        snapped = []
        for candidate in candidates:
            start, end = snap(candidate.start), snap(candidate.end)
            if end > start:
                snapped.append(ScoredSpan(start, end, candidate.score))
        if sample.cardinality == "multi":
            spans = consolidate_spans(snapped, sample.duration)
        else:
            spans = tuple(value for span in snapped[:1] if (value := span.clipped(sample.duration)))
        return Prediction(
            spans,
            raw,
            {
                "backend": self.name,
                "base_checkpoint": self.base_checkpoint,
                "adapter_checkpoint": self.adapter_checkpoint,
                "evidence_units": evidence.size,
                "visible_timestamps": len(visible),
                "encoder_pruning": self.encoder_pruning,
                "encoder_retention_ratio": self.encoder_retention,
                "post_pruning": self.post_pruning,
                "post_retention_ratio": self.post_retention,
                "max_new_tokens": max_new_tokens,
                "semvid": {key: value for key, value in evidence.metadata.items() if key.startswith("semvid_")},
                "timestamp_interleaved": True,
                "timestamp_snapping": True,
                "additional_training": False,
                "timing": {
                    "post_prune_seconds": pruning_finished - pruning_started,
                    "prompt_prefill_build_seconds": prompt_finished - pruning_finished,
                    "generation_seconds": generation_finished - prompt_finished,
                },
            },
        )


__all__ = [
    "UNITIME_EVIDENCE_BUDGET",
    "UniTimeEvidenceBackend",
    "_install_mage_qwen2_vision_pruning",
    "adaptive_frame_size",
    "compact_mrope_positions",
]
