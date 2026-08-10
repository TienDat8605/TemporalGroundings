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

    def prepare_inputs_for_generation(self, *args: Any, position_ids=None, cache_position=None, **kwargs: Any):
        values = super().prepare_inputs_for_generation(
            *args,
            position_ids=position_ids,
            cache_position=cache_position,
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


class QwenEvidenceBackend(ModelBackend):
    capabilities = frozenset({"encoded-evidence", "spatial-evidence", "generative"})

    def __init__(self, checkpoint: str, cache_dir: Path, *, name: str) -> None:
        self.name = name
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import AutoProcessor

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
        inputs = processor(text=[placeholder], videos=[frames], return_tensors="pt")
        device = self._device(model.model.visual)
        pixels = inputs["pixel_values_videos"].to(device)
        grids = inputs["video_grid_thw"].to(device)
        with torch.inference_mode():
            features, _ = model.model.get_video_features(pixels, grids)
        embedding = features[0]
        temporal, height, width = [int(value) for value in grids[0].tolist()]
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
        cardinality = "every disjoint interval" if sample.cardinality == "multi" else "the best interval"
        return (
            f"Find {cardinality} where this event occurs: {sample.query!r}. "
            f"Use seconds relative to this evidence window, from 0 to {context.duration:.3f}. "
            "Return only a JSON array of [start, end] pairs. Return [] if absent."
        )

    def _evidence_prompt(self, sample: Sample, evidence: TemporalEvidence, context: GroundingContext):
        import torch

        model, processor = self._load()
        tokenizer = processor.tokenizer
        groups: list[tuple[float, int]] = []
        for timestamp in evidence.timestamps:
            relative = max(0.0, timestamp - context.start)
            if groups and abs(groups[-1][0] - relative) < 1e-5:
                groups[-1] = (relative, groups[-1][1] + 1)
            else:
                groups.append((relative, 1))
        visual = "".join(
            f"<{timestamp:.1f} seconds>{processor.vision_start_token}"
            + processor.video_token * count
            + processor.vision_end_token
            for timestamp, count in groups
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
