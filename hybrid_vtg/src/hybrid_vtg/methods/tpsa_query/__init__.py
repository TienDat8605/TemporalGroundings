"""Query-aware post-encoder temporal/spatial evidence selection."""

from __future__ import annotations

import math
from pathlib import Path

from ...contracts import GroundingContext, Method, ModelBackend, Prediction, Sample
from ...media import uniform_timestamps


class TPSAQuery(Method):
    name = "tpsa-query"

    def __init__(self, retention_ratio: float = 0.125, fps: float = 2.0, maximum_frames: int = 768) -> None:
        if not 0 < retention_ratio <= 1:
            raise ValueError("retention ratio must be in (0, 1]")
        self.retention_ratio = retention_ratio
        self.fps = fps
        self.maximum_frames = maximum_frames

    @staticmethod
    def select_indices(scores, timestamps: tuple[float, ...], target: int):
        """Keep query peaks plus deterministic anchors across the complete timeline."""
        import torch

        scores = scores.float().flatten()
        if scores.numel() != len(timestamps):
            raise ValueError("query scores and evidence timestamps must align")
        target = min(max(1, target), scores.numel())
        if target == scores.numel():
            return torch.arange(target, device=scores.device)

        times = torch.tensor(timestamps, device=scores.device, dtype=torch.float32)
        band_count = min(target, max(1, math.ceil(math.sqrt(target))))
        edges = torch.linspace(times.min(), times.max() + 1e-6, band_count + 1, device=scores.device)
        selected: set[int] = set()
        for endpoint in (times.min(), times.max()):
            members = torch.nonzero(times == endpoint, as_tuple=False).flatten()
            selected.add(int(members[torch.argmax(scores[members])].item()))
        for band in range(band_count):
            members = torch.nonzero(
                (times >= edges[band]) & (times < edges[band + 1]),
                as_tuple=False,
            ).flatten()
            if members.numel():
                local = members[torch.argmax(scores[members])]
                selected.add(int(local.item()))
        ranked = torch.argsort(scores, descending=True, stable=True).tolist()
        for index in ranked:
            if len(selected) < target:
                selected.add(int(index))
            if len(selected) >= target:
                break
        ranked_selected = sorted(selected, key=lambda index: (-float(scores[index]), index))[:target]
        return torch.tensor(sorted(ranked_selected), device=scores.device, dtype=torch.long)

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        self.validate_model(model)
        frame_count = min(self.maximum_frames, max(2, math.ceil(sample.duration * self.fps)))
        if frame_count % 2:
            frame_count -= 1
        evidence = model.encode(sample, uniform_timestamps(0.0, sample.duration, frame_count))
        scores = model.query_scores(evidence, sample.query)
        target = max(1, round(evidence.size * self.retention_ratio))
        if model.maximum_evidence_units is not None:
            target = min(target, model.maximum_evidence_units)
        indices = self.select_indices(scores, evidence.timestamps, target)
        compact = evidence.select(indices.tolist())
        prediction = model.predict(sample, compact, GroundingContext(0.0, sample.duration))
        return Prediction(
            prediction.spans,
            prediction.raw_output,
            {
                **prediction.telemetry,
                "decoded_frames": frame_count,
                "original_evidence": evidence.size,
                "retained_evidence": compact.size,
                "retention_ratio": compact.size / evidence.size,
                "selected_indices": indices.cpu().tolist(),
                "post_encoder": True,
            },
        )
