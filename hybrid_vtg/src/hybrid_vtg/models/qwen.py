"""Direct Transformers adapter for Qwen3-VL and TimeLens2 evidence inference."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

from ..contracts import (
    GroundingContext,
    ModelBackend,
    Prediction,
    Sample,
    ScoredSpan,
    TemporalEvidence,
)
from ..media import extract_frames
from ..postprocess import consolidate_spans, parse_spans
from .pruning import mage_cell_plan, motion_residual_importance, semvid_select


def _attention_options() -> dict[str, str]:
    """Use FlashAttention 2 when its validated Transformers integration is available, else SDPA."""
    from transformers.utils import is_flash_attn_2_available

    if is_flash_attn_2_available():
        return {"attn_implementation": "flash_attention_2"}
    return {"attn_implementation": "sdpa"}


def _generation_token_budget(sample: Sample) -> int:
    """Allow OMTG to enumerate many occurrences without slowing other benchmarks."""
    return 256 if sample.group == "omtg" else 32


def _dense_evidence_units(metadata: dict[str, Any], fallback: int) -> int:
    """Recover the dense encoder count through method-level concatenation."""
    value = metadata.get("dense_evidence_units")
    if isinstance(value, int) and value > 0:
        return value
    parts = metadata.get("parts")
    if isinstance(parts, list):
        counts = [
            _dense_evidence_units(part, 0)
            for part in parts
            if isinstance(part, dict)
        ]
        if any(counts):
            return sum(counts)
    return fallback


class _CompactQwenMixin:
    """Preserve explicit compact-prefill positions on the first generation step."""

    def prepare_inputs_for_generation(
        self,
        *args: Any,
        position_ids=None,
        cache_position=None,
        inputs_embeds=None,
        **kwargs: Any,
    ):
        # `inputs_embeds` must be an explicit named parameter, not hidden in **kwargs:
        # transformers/generation/utils.py inspects the *signature* of
        # prepare_inputs_for_generation for the parameter name before accepting
        # inputs_embeds in .generate().
        values = super().prepare_inputs_for_generation(
            *args,
            position_ids=position_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if position_ids is not None and cache_position is not None and int(cache_position[0]) == 0:
            values["position_ids"] = position_ids
        return values


def _model_class():
    from transformers import Qwen3VLForConditionalGeneration

    class CompactQwen(_CompactQwenMixin, Qwen3VLForConditionalGeneration):
        pass

    return CompactQwen


def _install_mage_vision_pruning(model, prune_layer: int) -> None:
    """Install Mage-style motion/residual pruning before a Qwen vision block.

    The selection plan is prepared per encoded video from processed-grid-aligned
    motion/residual maps. Original rotary coordinates are sliced rather than
    compressed, and variable per-time sequence lengths are passed explicitly.
    """
    import torch
    import torch.nn.functional as functional

    visual = model.model.visual
    merge = int(model.config.vision_config.spatial_merge_size)
    original_forward = visual.forward
    original_get_video_features = model.model.get_video_features

    if not 0 <= prune_layer < len(visual.blocks):
        raise ValueError(f"encoder prune layer must be between 0 and {len(visual.blocks) - 1}")

    def sparse_forward(hidden_states, grid_thw, **kwargs):
        plan = getattr(model.model, "_mage_prune_plan", None)
        if plan is None:
            return original_forward(hidden_states, grid_thw, **kwargs)
        if int(grid_thw.shape[0]) != 1:
            raise ValueError("Mage-style encoder pruning currently supports batch size one")
        hidden_states = visual.patch_embed(hidden_states)
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = functional.pad(cu_seqlens, (1, 0), value=0)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(visual.blocks):
            if layer_num == prune_layer:
                index = torch.tensor(plan.patch_indices, device=hidden_states.device, dtype=torch.long)
                hidden_states = hidden_states[index]
                per_time = torch.tensor(
                    [value * merge * merge for value in plan.cells_per_time],
                    device=grid_thw.device,
                    dtype=torch.int32,
                )
                cu_seqlens = per_time.cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = functional.pad(cu_seqlens, (1, 0), value=0)
                cos, sin = position_embeddings
                position_embeddings = (cos[index], sin[index])
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in visual.deepstack_visual_indexes:
                deepstack_feature = visual.deepstack_merger_list[
                    visual.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = visual.merger(hidden_states)
        model.model._mage_output_counts = [plan.target_cells]
        return hidden_states, deepstack_feature_lists

    visual.forward = sparse_forward

    def sparse_get_video_features(pixel_values_videos, video_grid_thw):
        plan = getattr(model.model, "_mage_prune_plan", None)
        if plan is None:
            return original_get_video_features(pixel_values_videos, video_grid_thw)
        pixel_values_videos = pixel_values_videos.type(visual.dtype)
        video_embeds, deepstack_video_embeds = visual(pixel_values_videos, grid_thw=video_grid_thw)
        counts = getattr(model.model, "_mage_output_counts", [plan.target_cells])
        return torch.split(video_embeds, counts), deepstack_video_embeds

    model.model.get_video_features = sparse_get_video_features


class QwenEvidenceBackend(ModelBackend):
    capabilities = frozenset(
        {"encoded-evidence", "spatial-evidence", "generative", "timestamp-interleaved", "native-video-grounding"}
    )

    def __init__(
        self,
        checkpoint: str,
        cache_dir: Path,
        *,
        name: str,
        encoder_pruning: str = "none",
        encoder_retention: float = 1.0,
        encoder_prune_layer: int = 0,
        post_pruning: str = "none",
        post_retention: float = 1.0,
        maximum_evidence_units: int | None = 4_096,
    ) -> None:
        if encoder_pruning not in {"none", "mage"}:
            raise ValueError("encoder pruning must be 'none' or 'mage'")
        if post_pruning not in {"none", "semvid"}:
            raise ValueError("post pruning must be 'none' or 'semvid'")
        if not 0 < encoder_retention <= 1 or not 0 < post_retention <= 1:
            raise ValueError("pruning retention ratios must be in (0, 1]")
        if encoder_prune_layer < 0:
            raise ValueError("encoder prune layer must be non-negative")
        if encoder_pruning == "none" and encoder_retention != 1.0:
            raise ValueError("encoder retention requires --encoder-pruning mage")
        if post_pruning == "none" and post_retention != 1.0:
            raise ValueError("post retention requires --post-pruning semvid")
        if encoder_pruning == "mage" and post_pruning == "semvid" and post_retention > encoder_retention:
            raise ValueError("post retention cannot exceed encoder retention when both policies are enabled")
        if maximum_evidence_units is not None and maximum_evidence_units <= 0:
            raise ValueError("maximum evidence units must be positive")
        self.name = name
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self.encoder_pruning = encoder_pruning
        self.encoder_retention = encoder_retention
        self.encoder_prune_layer = encoder_prune_layer
        self.post_pruning = post_pruning
        self.post_retention = post_retention
        self._maximum_evidence_units = maximum_evidence_units
        self._model: Any = None
        self._processor: Any = None

    @property
    def maximum_evidence_units(self) -> int | None:
        return self._maximum_evidence_units

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import AutoProcessor
            from transformers import logging as hf_logging

            # Suppress the large model-weight / device-map load reports. These can be
            # several screens of output for a 4B checkpoint and bury the run progress.
            previous = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                import torch

                self._processor = AutoProcessor.from_pretrained(self.checkpoint)
                target_dtype = torch.float16 if torch.cuda.is_available() else "auto"
                self._model = (
                    _model_class()
                    .from_pretrained(
                        self.checkpoint,
                        torch_dtype=target_dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        **_attention_options(),
                    )
                    .eval()
                )
                if self.encoder_pruning == "mage":
                    _install_mage_vision_pruning(self._model, self.encoder_prune_layer)
            finally:
                hf_logging.set_verbosity(previous)
        return self._model, self._processor

    @staticmethod
    def _device(module: Any):
        return next(module.parameters()).device

    def encode(self, sample: Sample, timestamps: Sequence[float]) -> TemporalEvidence:
        import numpy as np
        import torch
        from PIL import Image

        model, processor = self._load()
        paths = extract_frames(
            sample.video_path,
            timestamps,
            self.cache_dir / "frames",
            maximum_width=2_048 if self.maximum_evidence_units is not None else 336,
        )
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        placeholder = processor.vision_start_token + processor.video_token + processor.vision_end_token
        do_resize = True
        if self.maximum_evidence_units is not None:
            temporal_patch = int(model.config.vision_config.temporal_patch_size)
            patch = int(model.config.vision_config.patch_size)
            merge = int(model.config.vision_config.spatial_merge_size)
            temporal = max(1, math.ceil(len(frames) / temporal_patch))
            cells_per_time = max(1, self.maximum_evidence_units // temporal)
            aspect = frames[0].width / frames[0].height
            cell_height = max(1, round(math.sqrt(cells_per_time / aspect)))
            cell_width = max(1, cells_per_time // cell_height)
            while cell_height * cell_width > cells_per_time:
                if cell_width >= cell_height:
                    cell_width -= 1
                else:
                    cell_height -= 1
            alignment = patch * merge
            size = (cell_width * alignment, cell_height * alignment)
            frames = [frame.resize(size, Image.Resampling.LANCZOS) for frame in frames]
            do_resize = False
        # TimeLens2 ships video_processor.do_resize=False, which leaves frames at their
        # native resolution while the grid is computed from a resized size, so the
        # patch reshape mismatches. Force resizing so grid and tensor agree.
        inputs = processor(
            text=[placeholder],
            videos=[frames],
            return_tensors="pt",
            do_resize=do_resize,
        )
        device = self._device(model.model.visual)
        pixels = inputs["pixel_values_videos"].to(device)
        grids = inputs["video_grid_thw"].to(device)
        grid = grids[0]
        temporal, height, width = [int(value) for value in grid.tolist()]
        merge = int(model.config.vision_config.spatial_merge_size)
        dense_per_time = height * width // (merge * merge)
        plan = None
        importance_cache_hit = False
        if self.encoder_pruning == "mage":
            stat = sample.video_path.stat()
            cache_identity = "|".join(
                [
                    str(sample.video_path.resolve()),
                    str(stat.st_size),
                    str(stat.st_mtime_ns),
                    ",".join(f"{value:.6f}" for value in timestamps),
                    f"{temporal}x{height // merge}x{width // merge}",
                ]
            )
            digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
            importance_path = self.cache_dir / "mage-maps" / f"{digest}.npz"
            try:
                with np.load(importance_path, allow_pickle=False) as cached:
                    importance = cached["importance"]
                expected = (temporal, height // merge, width // merge)
                if importance.shape != expected:
                    raise ValueError(f"cached importance has shape {importance.shape}, expected {expected}")
                importance_cache_hit = True
            except (FileNotFoundError, KeyError, OSError, ValueError):
                importance = motion_residual_importance(
                    frames,
                    temporal_units=temporal,
                    cell_height=height // merge,
                    cell_width=width // merge,
                )
                importance_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = importance_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, importance=importance)
                temporary.replace(importance_path)
            plan = mage_cell_plan(
                importance,
                merge_size=merge,
                retention_ratio=self.encoder_retention,
            )
            model.model._mage_prune_plan = plan
        try:
            with torch.inference_mode():
                features, _ = model.model.get_video_features(pixels, grids)
        finally:
            if plan is not None:
                model.model._mage_prune_plan = None
                model.model._mage_output_counts = None
        embedding = features[0]
        cells_per_time = plan.cells_per_time if plan is not None else (dense_per_time,) * temporal
        if sum(cells_per_time) != int(embedding.shape[0]):
            raise RuntimeError("Qwen vision features do not match video_grid_thw")
        boundaries = np.linspace(0, len(timestamps), temporal + 1).round().astype(int)
        unit_times = []
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
        return TemporalEvidence(
            embeddings=embedding,
            timestamps=tuple(unit_times),
            source_frames=len(timestamps),
            metadata={
                "backend": self.name,
                "grid_thw": [temporal, height, width],
                "tokens_per_time": list(cells_per_time),
                "cell_coordinates": [list(value) for value in coordinates],
                "dense_evidence_units": temporal * dense_per_time,
                "maximum_evidence_units": self.maximum_evidence_units,
                "encoder_pruning": self.encoder_pruning,
                "encoder_retention_ratio": self.encoder_retention,
                "encoder_prune_layer": self.encoder_prune_layer,
                "encoder_retained_evidence_units": embedding.shape[0],
                "importance_backend": "optical-flow-motion-compensated-residual" if plan is not None else None,
                "importance_cache_hit": importance_cache_hit if plan is not None else None,
                "frame_paths": [str(path) for path in paths],
            },
        )

    def _query_embeddings(self, query: str):
        import torch
        import torch.nn.functional as functional

        model, processor = self._load()
        device = self._device(model.model.get_input_embeddings())
        tokens = processor.tokenizer(query, return_tensors="pt", add_special_tokens=False)
        ids = tokens["input_ids"].to(device)
        with torch.inference_mode():
            embedding = model.model.get_input_embeddings()(ids)[0].float()
        return functional.normalize(embedding, dim=-1, eps=1e-6)

    def query_scores(self, evidence: TemporalEvidence, query: str):
        import torch
        import torch.nn.functional as functional

        visual = functional.normalize(evidence.embeddings.float(), dim=-1, eps=1e-6)
        text = self._query_embeddings(query).to(visual.device)
        return torch.matmul(visual, text.T).amax(dim=-1)

    def _prompt(self, sample: Sample, context: GroundingContext) -> str:
        if self.name == "timelens2-4b":
            return (
                f'Given the query: "{sample.query}", return ALL time spans (in seconds) where the query is relevant.\n'
                "Output format MUST be a JSON array of [start, end] pairs.\n"
            )
        if sample.cardinality == "multi":
            return (
                f"The event '{sample.query}' may occur MULTIPLE times in this video. "
                f"List EVERY occurrence as its own [start, end] pair, in chronological order. "
                "Keep nearby but distinct occurrences as separate pairs; do not merge them. "
                f"Use seconds relative to this evidence window, from 0 to {context.duration:.3f}. "
                "Return ONLY a JSON array of [start, end] pairs, e.g. [[1.0, 3.0], [7.5, 9.0]]. "
                f"Do NOT return the whole video as a single pair like [0, {context.duration:.3f}]. "
                "Return [] if the event never occurs."
            )
        return (
            f"Find the best interval where this event occurs: {sample.query!r}. "
            f"Use seconds relative to this evidence window, from 0 to {context.duration:.3f}. "
            "Return only a JSON array of [start, end] pairs. Return [] if absent."
        )

    def _evidence_prompt(self, sample: Sample, evidence: TemporalEvidence, context: GroundingContext):
        import torch

        model, processor = self._load()
        tokenizer = processor.tokenizer
        groups: list[tuple[float, int]] = []
        for index, timestamp in enumerate(evidence.timestamps):
            relative = max(0.0, timestamp - context.start)
            if groups and abs(groups[-1][0] - relative) < 1e-5:
                groups[-1] = (relative, groups[-1][1] + 1)
            else:
                groups.append((relative, 1))
        visual = "".join(
            f"<{timestamp:.2f} seconds>{processor.vision_start_token}"
            + processor.video_token * count
            + processor.vision_end_token
            for timestamp, count in groups
        )
        prompt_text = self._prompt(sample, context)
        if self.name == "timelens2-4b":
            text = f"<|im_start|>user\n{visual}{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            text = f"<|im_start|>user\n{prompt_text}\n{visual}<|im_end|>\n<|im_start|>assistant\n"
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        positions = (input_ids == processor.video_token_id).nonzero(as_tuple=False)[:, 1]
        if positions.numel() != evidence.size:
            raise RuntimeError(f"prompt has {positions.numel()} visual slots for {evidence.size} evidence rows")
        device = self._device(model.model.get_input_embeddings())
        input_ids = input_ids.to(device)
        attention = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            inputs_embeds = model.model.get_input_embeddings()(input_ids)
            inputs_embeds[0, positions.to(device)] = evidence.embeddings.to(
                device=device,
                dtype=inputs_embeds.dtype,
            )
        position_ids = torch.arange(input_ids.shape[1], device=device).view(1, 1, -1).expand(3, 1, -1)
        model.model.rope_deltas = torch.zeros((1, 1), device=device, dtype=torch.long)
        return input_ids, attention, inputs_embeds, position_ids

    def predict(
        self,
        sample: Sample,
        evidence: TemporalEvidence,
        context: GroundingContext,
    ) -> Prediction:
        import torch

        model, processor = self._load()
        if self.post_pruning == "semvid":
            evidence = semvid_select(
                evidence,
                self._query_embeddings(sample.query),
                retention_ratio=self.post_retention,
                dense_evidence_units=_dense_evidence_units(evidence.metadata, evidence.size),
            )
        input_ids, attention, inputs_embeds, position_ids = self._evidence_prompt(
            sample,
            evidence,
            context,
        )
        max_new_tokens = _generation_token_budget(sample)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        output_ids = generated[:, input_ids.shape[1] :] if generated.shape[1] > input_ids.shape[1] else generated
        raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        local = parse_spans(raw)
        global_spans = tuple(
            ScoredSpan(span.start + context.start, span.end + context.start, span.score) for span in local
        )
        if sample.cardinality == "multi":
            spans = consolidate_spans(global_spans, sample.duration)
        else:
            spans = tuple(value for span in global_spans if (value := span.clipped(sample.duration)))
        return Prediction(
            spans,
            raw,
            {
                "backend": self.name,
                "checkpoint": self.checkpoint,
                "evidence_units": evidence.size,
                "encoder_pruning": self.encoder_pruning,
                "encoder_retention_ratio": self.encoder_retention,
                "post_pruning": self.post_pruning,
                "post_retention_ratio": self.post_retention,
                "max_new_tokens": max_new_tokens,
                "llm_input_tokens": int(input_ids.shape[1]),
                "llm_output_tokens": int(output_ids.shape[1]),
                "semvid": {
                    key: value for key, value in evidence.metadata.items() if key.startswith("semvid_")
                },
                "context": {"start": context.start, "end": context.end},
            },
        )

    def predict_video(self, sample: Sample) -> Prediction:
        """Run the official TimeLens2 whole-video control path."""
        if self.name not in {"timelens2-4b", "timelens-8b"}:
            raise ValueError("native video inference is available only for a TimeLens checkpoint")
        from .timelens import native_timelens_prediction, require_native_video_reader

        require_native_video_reader()
        model, processor = self._load()
        return native_timelens_prediction(model, processor, sample, family="qwen3")
