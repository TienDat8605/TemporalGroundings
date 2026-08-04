"""Architecture boundary for sparse frozen Qwen3-VL visual inference.

The stock Transformers generation path couples visual embeddings, placeholder
tokens, multimodal rotary positions, and deep-stack features. This adapter
validates and packages a sparse selection, but only calls models exposing the
explicit ``generate_from_sparse_visual`` interface. It never silently falls
back to zeroing tokens because that would not reduce sequence cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class SparseAdapterCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class SparseVisualInputs:
    visual_features: torch.Tensor
    selected_positions: torch.Tensor
    original_positions: torch.Tensor
    frame_indices: torch.Tensor
    grid_yx: torch.Tensor
    original_token_count: int

    @property
    def retained_token_count(self) -> int:
        return int(self.visual_features.shape[0])


@dataclass(frozen=True)
class SparseAdapterAudit:
    compatible: bool
    dense_token_count: int
    retained_token_count: int
    placeholder_count: int
    has_deepstack: bool
    reason: str


def grid_coordinates(grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError('grid_thw must have shape (num_videos, 3)')
    frames = []
    coordinates = []
    frame_offset = 0
    for temporal, height, width in grid_thw.detach().cpu().tolist():
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError('visual grid is not divisible by spatial_merge_size')
        merged_h, merged_w = height // spatial_merge_size, width // spatial_merge_size
        for local_frame in range(temporal):
            for y in range(merged_h):
                for x in range(merged_w):
                    frames.append(frame_offset + local_frame)
                    coordinates.append((y, x))
        frame_offset += temporal
    return torch.tensor(frames, dtype=torch.long), torch.tensor(coordinates, dtype=torch.long)


class QwenSparseAdapter:
    def __init__(self, model: Any):
        self.model = model

    def audit(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
        selected_mask: torch.Tensor,
        *,
        deepstack_visual: list[torch.Tensor] | None = None,
    ) -> SparseAdapterAudit:
        if visual_features.ndim != 2:
            raise ValueError('visual_features must have shape (tokens, hidden)')
        selected = selected_mask.reshape(-1).to(dtype=torch.bool, device=visual_features.device)
        if len(selected) != len(visual_features):
            raise ValueError('selected_mask must match visual token count')
        config = getattr(self.model, 'config', None)
        video_token_id = getattr(config, 'video_token_id', None)
        if video_token_id is None:
            return SparseAdapterAudit(False, len(visual_features), int(selected.sum()), 0,
                                      bool(deepstack_visual), 'model has no video_token_id')
        placeholders = int((input_ids == video_token_id).sum().item())
        if placeholders != len(visual_features):
            return SparseAdapterAudit(False, len(visual_features), int(selected.sum()), placeholders,
                                      bool(deepstack_visual), 'placeholder/feature cardinality mismatch')
        if deepstack_visual and any(len(value) != len(visual_features) for value in deepstack_visual):
            return SparseAdapterAudit(False, len(visual_features), int(selected.sum()), placeholders,
                                      True, 'deep-stack cardinality mismatch')
        interface = callable(getattr(self.model, 'generate_from_sparse_visual', None))
        reason = 'compatible sparse-generation interface' if interface else (
            'stock Qwen generation has no sparse visual interface; use proposal scorer'
        )
        return SparseAdapterAudit(interface, len(visual_features), int(selected.sum()), placeholders,
                                  bool(deepstack_visual), reason)

    def prepare(
        self,
        visual_features: torch.Tensor,
        selected_mask: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        spatial_merge_size: int,
    ) -> SparseVisualInputs:
        selected = selected_mask.reshape(-1).to(dtype=torch.bool, device=visual_features.device)
        if len(selected) != len(visual_features):
            raise ValueError('selected mask and visual features differ')
        frame_indices, grid_yx = grid_coordinates(grid_thw, spatial_merge_size)
        if len(frame_indices) != len(visual_features):
            raise ValueError('grid metadata and visual features differ')
        positions = torch.arange(len(visual_features), device=visual_features.device)
        return SparseVisualInputs(
            visual_features=visual_features[selected],
            selected_positions=positions[selected],
            original_positions=positions,
            frame_indices=frame_indices.to(visual_features.device)[selected],
            grid_yx=grid_yx.to(visual_features.device)[selected],
            original_token_count=len(visual_features),
        )

    @torch.inference_mode()
    def generate(self, sparse: SparseVisualInputs, **generation_inputs):
        interface = getattr(self.model, 'generate_from_sparse_visual', None)
        if not callable(interface):
            raise SparseAdapterCompatibilityError(
                'This checkpoint does not expose generate_from_sparse_visual; '
                'use the analytic proposal scorer instead of claiming sparse VideoLLM inference.'
            )
        return interface(sparse_visual=sparse, **generation_inputs)
