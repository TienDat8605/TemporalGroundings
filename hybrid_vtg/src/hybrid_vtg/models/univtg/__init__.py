"""Frozen inference adapter for official UniVTG moment-retrieval checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ...contracts import (
    GroundingContext,
    ModelBackend,
    Prediction,
    Sample,
    ScoredSpan,
    TemporalEvidence,
)
from ...postprocess import temporal_nms
from .features import UniVTGFeatures
from .vendor.network import NetworkSpec, UniVTGNetwork

FEATURE_STACKS = {
    "clip-b16": 512,
    "clip-b32": 512,
    "slowfast-clip-b32": 2816,
}


def _infer_network_spec(state: dict[str, Any], maximum_video_length: int = 75) -> NetworkSpec:
    video_dim = int(state["input_vid_proj.0.LayerNorm.weight"].numel())
    text_dim = int(state["input_txt_proj.0.LayerNorm.weight"].numel())
    hidden = int(state["token_type_embeddings.weight"].shape[1])
    layers = 1 + max(int(key.split(".")[3]) for key in state if key.startswith("transformer.encoder.layers."))
    projections = len({key.split(".")[1] for key in state if key.startswith("input_vid_proj.")})
    return NetworkSpec(
        video_dim=video_dim,
        text_dim=text_dim,
        hidden_dim=hidden,
        encoder_layers=layers,
        input_projections=projections,
        text_positions=False,
        maximum_video_length=maximum_video_length,
    )


class UniVTGBackend(ModelBackend):
    name = "univtg"
    capabilities = frozenset({"encoded-evidence", "temporal-evidence", "dense-proposals"})

    def __init__(
        self,
        checkpoint: str | None,
        cache_dir: Path,
        model_spec: str | None = None,
        feature_roots: tuple[Path, ...] = (),
    ) -> None:
        if not checkpoint:
            raise ValueError("--checkpoint is required for UniVTG")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.cache_dir = cache_dir
        self._requested_feature_stack = model_spec
        self._feature_roots = feature_roots
        self._network = None
        self._features = None
        self._spec = None
        self._load()

    def _sidecar(self) -> dict[str, Any]:
        for name in ("opt.json", "config.json"):
            path = self.checkpoint.parent / name
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _feature_stack(self, raw_dim: int, sidecar: dict[str, Any]) -> str:
        requested = self._requested_feature_stack
        if requested:
            if requested not in FEATURE_STACKS:
                raise ValueError(f"unknown UniVTG feature stack {requested!r}")
            return requested
        types = sidecar.get("v_feat_types")
        if isinstance(types, list):
            types = "_".join(types)
        if types and "slowfast" in str(types):
            return "slowfast-clip-b32"
        if raw_dim == 2816:
            return "slowfast-clip-b32"
        if "b16" in str(sidecar).lower():
            return "clip-b16"
        return "clip-b32"

    def _load(self) -> None:
        import torch

        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        embedded = checkpoint.get("opt") if isinstance(checkpoint, dict) else None
        embedded_options = vars(embedded) if embedded is not None and hasattr(embedded, "__dict__") else {}
        sidecar = {**embedded_options, **self._sidecar()}
        maximum = int(sidecar.get("max_v_l", 75))
        spec = _infer_network_spec(state, maximum)
        if sidecar.get("use_txt_pos"):
            from dataclasses import replace

            spec = replace(spec, text_positions=True)
        raw_dim = spec.video_dim - 2 if spec.video_dim - 2 in FEATURE_STACKS.values() else spec.video_dim
        stack = self._feature_stack(raw_dim, sidecar)
        expected = FEATURE_STACKS[stack]
        if raw_dim != expected:
            raise ValueError(f"checkpoint expects {raw_dim} raw video features, but {stack} provides {expected}")
        network = UniVTGNetwork(spec)
        network.load_state_dict(state, strict=True)
        network.eval()
        if torch.cuda.is_available():
            network.cuda()
        self._network = network
        self._spec = spec
        self._uses_tef = spec.video_dim == raw_dim + 2
        self._features = UniVTGFeatures(self.cache_dir, stack, self._feature_roots)
        self.feature_stack = stack

    @property
    def maximum_evidence_units(self) -> int | None:
        return self._spec.maximum_video_length

    def encode(self, sample: Sample, timestamps: Sequence[float]) -> TemporalEvidence:
        import torch

        values = self._features.video(sample, timestamps)
        expected = self._spec.video_dim - (2 if self._uses_tef else 0)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(
                f"UniVTG checkpoint expects {expected} feature values per timestamp, "
                f"but the configured feature source produced shape {values.shape}"
            )
        return TemporalEvidence(
            embeddings=torch.from_numpy(values).to(next(self._network.parameters()).device),
            timestamps=tuple(float(value) for value in timestamps),
            source_frames=len(timestamps),
            metadata={"backend": self.name, "feature_stack": self.feature_stack},
        )

    def query_scores(self, evidence: TemporalEvidence, query: str):
        import torch
        import torch.nn.functional as functional

        text = self._features.text(query).to(evidence.embeddings.device)
        sentence = functional.normalize(text.mean(dim=0), dim=0, eps=1e-6)
        clip = functional.normalize(evidence.embeddings[..., -512:].float(), dim=-1, eps=1e-6)
        return torch.matmul(clip, sentence)

    def predict(
        self,
        sample: Sample,
        evidence: TemporalEvidence,
        context: GroundingContext,
    ) -> Prediction:
        import torch

        if evidence.size > self.maximum_evidence_units:
            raise ValueError(f"UniVTG checkpoint accepts at most {self.maximum_evidence_units} evidence units")
        device = next(self._network.parameters()).device
        video = evidence.embeddings.float().to(device)
        relative = torch.tensor(
            [(value - context.start) / context.duration for value in evidence.timestamps],
            device=device,
        ).clamp(0, 1)
        if self._uses_tef:
            half = 0.5 / max(evidence.size, 1)
            tef = torch.stack(((relative - half).clamp(0, 1), (relative + half).clamp(0, 1)), dim=-1)
            video = torch.cat((video, tef), dim=-1)
        text = self._features.text(sample.query).to(device)
        with torch.inference_mode():
            output = self._network(
                text.unsqueeze(0),
                torch.ones((1, text.shape[0]), device=device),
                video.unsqueeze(0),
                torch.ones((1, video.shape[0]), device=device),
                video_positions=relative.unsqueeze(0),
            )
        scores = output["pred_logits"][0, :, 0]
        normalized = relative[:, None] + output["pred_spans"][0]
        spans = [
            ScoredSpan(
                context.start + float(start.clamp(0, 1)) * context.duration,
                context.start + float(end.clamp(0, 1)) * context.duration,
                float(score),
            )
            for (start, end), score in zip(normalized, scores)
            if float(end) > float(start)
        ]
        ranked = temporal_nms(spans, threshold=0.7, maximum=100)
        if sample.cardinality == "multi":
            selected = tuple(value for value in ranked if value.score >= 0.5) or ranked[:1]
        else:
            selected = ranked[:10]
        return Prediction(
            selected,
            "",
            {
                "backend": self.name,
                "checkpoint": str(self.checkpoint),
                "feature_stack": self.feature_stack,
                "evidence_units": evidence.size,
                "nms_threshold": 0.7,
                "weights_frozen": True,
            },
        )


__all__ = ["UniVTGBackend"]
