"""Direct Transformers adapter for Qwen3-VL and TimeLens2 evidence inference."""

from __future__ import annotations

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


def _prune_indices(grid_thw, merge: int, prune_ratio: float) -> tuple[list[int], list[list[int]]]:
    """Indices of surviving patch tokens after dropping whole 2x2 merge cells.

    Keeps a rectangular sub-grid of merge cells so the ``Qwen3VLVisionPatchMerger``
    reshape invariant (4 consecutive tokens per cell) is preserved. ``prune_ratio``
    is the fraction of cells retained:

    - ``0.5``  -> keep even cell rows (halve height)
    - ``0.25`` -> keep even cell rows and columns (halve both dims)

    Returns ``(survive_indices, new_grid)`` where ``new_grid`` is the reduced
    ``[t, h, w]`` per frame, matching the surviving token count.
    """
    if prune_ratio == 0.5:
        step_r, step_c = 2, 1
    elif prune_ratio == 0.25:
        step_r, step_c = 2, 2
    else:
        raise ValueError(f"unsupported prune_ratio {prune_ratio!r}; use 0.5 or 0.25")
    survive: list[int] = []
    new_grid: list[list[int]] = []
    offset = 0
    for t, h, w in grid_thw.tolist():
        cells_h, cells_w = h // merge, w // merge
        kept_rows = (cells_h + step_r - 1) // step_r
        kept_cols = (cells_w + step_c - 1) // step_c
        # A video row stacks t temporal frames; the same spatial cell pattern is
        # kept in every frame, so repeat it t times.
        for frame in range(t):
            frame_base = offset + frame * h * w
            for cell_row in range(0, cells_h, step_r):
                for cell_col in range(0, cells_w, step_c):
                    cell_index = cell_row * cells_w + cell_col
                    start = frame_base + cell_index * merge * merge
                    survive.extend(range(start, start + merge * merge))
        offset += t * h * w
        # Reduced grid must match the actual kept cell count so cu_seqlens and the
        # merger reshape stay consistent with the pruned token sequence.
        new_grid.append([t, kept_rows * merge, kept_cols * merge])
    return survive, new_grid


def _install_vision_pruning(model, prune_ratio: float, prune_layer: int) -> None:
    """Wrap the Qwen vision encoder to drop spatial merge cells mid-forward.

    Prunes after ``prune_layer`` blocks, then recomputes ``cu_seqlens`` and the
    rotary ``position_embeddings`` for the surviving tokens and returns a reduced
    ``grid_thw`` so downstream ``split_sizes`` stay consistent. DeepStack features
    are computed on the pruned sequence and are discarded by the backend, so their
    alignment is not a correctness concern.
    """
    import torch
    import torch.nn.functional as functional

    visual = model.model.visual
    merge = int(model.config.vision_config.spatial_merge_size)
    original_forward = visual.forward

    def pruned_forward(hidden_states, grid_thw, **kwargs):
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
                survive, new_grid = _prune_indices(grid_thw, merge, prune_ratio)
                index = torch.tensor(survive, device=hidden_states.device, dtype=torch.long)
                hidden_states = hidden_states[index]
                grid_thw = torch.tensor(new_grid, device=grid_thw.device, dtype=grid_thw.dtype)
                cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = functional.pad(cu_seqlens, (1, 0), value=0)
                cos, sin = position_embeddings
                position_embeddings = (cos[index], sin[index])
                model.model._pruned_grid_thw = grid_thw[0]
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
        return hidden_states, deepstack_feature_lists, grid_thw

    visual.forward = pruned_forward

    def pruned_get_image_features(pixel_values, image_grid_thw):
        image_embeds, deepstack_image_embeds, pruned_grid = visual(
            pixel_values, grid_thw=image_grid_thw
        )
        split_sizes = (pruned_grid.prod(-1) // merge**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    model.model.get_image_features = pruned_get_image_features
    del original_forward


class QwenEvidenceBackend(ModelBackend):
    capabilities = frozenset({"encoded-evidence", "spatial-evidence", "generative"})

    def __init__(
        self,
        checkpoint: str,
        cache_dir: Path,
        *,
        name: str,
        prune_ratio: float = 0.0,
        prune_layer: int = 12,
    ) -> None:
        self.name = name
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self.prune_ratio = prune_ratio
        self.prune_layer = prune_layer
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import AutoProcessor
            from transformers import logging as hf_logging

            # Suppress the large model-weight / device-map load reports. These can be
            # several screens of output for a 4B checkpoint and bury the run progress.
            previous = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                self._processor = AutoProcessor.from_pretrained(self.checkpoint)
                self._model = (
                    _model_class()
                    .from_pretrained(
                        self.checkpoint,
                        torch_dtype="auto",
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    .eval()
                )
                if self.prune_ratio:
                    _install_vision_pruning(self._model, self.prune_ratio, self.prune_layer)
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
        paths = extract_frames(sample.video_path, timestamps, self.cache_dir / "frames")
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        placeholder = processor.vision_start_token + processor.video_token + processor.vision_end_token
        # TimeLens2 ships video_processor.do_resize=False, which leaves frames at their
        # native resolution while the grid is computed from a resized size, so the
        # patch reshape mismatches. Force resizing so grid and tensor agree.
        inputs = processor(
            text=[placeholder],
            videos=[frames],
            return_tensors="pt",
            do_resize=True,
        )
        device = self._device(model.model.visual)
        pixels = inputs["pixel_values_videos"].to(device)
        grids = inputs["video_grid_thw"].to(device)
        with torch.inference_mode():
            features, _ = model.model.get_video_features(pixels, grids)
        embedding = features[0]
        pruned = getattr(model.model, "_pruned_grid_thw", None)
        grid = pruned if pruned is not None else grids[0]
        temporal, height, width = [int(value) for value in grid.tolist()]
        merge = int(model.config.vision_config.spatial_merge_size)
        per_time = height * width // (merge * merge)
        if temporal * per_time != int(embedding.shape[0]):
            raise RuntimeError("Qwen vision features do not match video_grid_thw")
        boundaries = np.linspace(0, len(timestamps), temporal + 1).round().astype(int)
        unit_times = []
        for index in range(temporal):
            values = timestamps[boundaries[index] : max(boundaries[index] + 1, boundaries[index + 1])]
            center = sum(values) / len(values) if values else timestamps[min(index, len(timestamps) - 1)]
            unit_times.extend([float(center)] * per_time)
        return TemporalEvidence(
            embeddings=embedding,
            timestamps=tuple(unit_times),
            source_frames=len(timestamps),
            metadata={
                "backend": self.name,
                "grid_thw": [temporal, height, width],
                "tokens_per_time": per_time,
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

    @staticmethod
    def _prompt(sample: Sample, context: GroundingContext) -> str:
        if sample.cardinality == "multi":
            return (
                f"The event '{sample.query}' may occur MULTIPLE times in this video. "
                f"List EVERY occurrence as its own [start, end] pair, in chronological order. "
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
        groups: list[tuple[float, int, str]] = []
        for index, timestamp in enumerate(evidence.timestamps):
            relative = max(0.0, timestamp - context.start)
            role = evidence.roles[index] if evidence.roles else ""
            if groups and abs(groups[-1][0] - relative) < 1e-5:
                groups[-1] = (relative, groups[-1][1] + 1, groups[-1][2])
            else:
                groups.append((relative, 1, role))
        visual = "".join(
            f"<{timestamp:.2f} seconds>{f', {role}' if role else ''}{processor.vision_start_token}"
            + processor.video_token * count
            + processor.vision_end_token
            for timestamp, count, role in groups
        )
        text = f"<|im_start|>user\n{self._prompt(sample, context)}\n{visual}<|im_end|>\n<|im_start|>assistant\n"
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
        input_ids, attention, inputs_embeds, position_ids = self._evidence_prompt(
            sample,
            evidence,
            context,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                position_ids=position_ids,
                max_new_tokens=256,
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
                "context": {"start": context.start, "end": context.end},
            },
        )
