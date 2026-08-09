"""TPSA selector on top of SemVID's audited Qwen compact-prefill path."""

from typing import Optional

import torch

from src.models.models.modeling_qwen3_vl_semvid import Qwen3VLForConditionalGenerationSemVID

from .config import SpatialAllocatorConfig
from .tpsa import TimelinePreservingSpatialAllocator, effective_temporal_fps


class Qwen3VLForConditionalGenerationTPSA(Qwen3VLForConditionalGenerationSemVID):
    """Override only visual selection; all compact-prefill mechanics stay inherited."""

    def __init__(self, config):
        super().__init__(config)
        allocator_config = SpatialAllocatorConfig(
            spatial_policy=getattr(config, "spatial_policy", "tpsa_boundary"),
            retention_ratio=float(getattr(config, "semantic_retention_ratio", 0.125)),
            mmr_lambda=float(getattr(config, "tpsa_mmr_lambda", 0.9)),
            relevance_top_fraction=float(getattr(config, "tpsa_relevance_top_fraction", 0.10)),
            boundary_quota_fraction=float(getattr(config, "tpsa_boundary_quota_fraction", 0.10)),
            motion_neighborhood_radius=int(getattr(config, "tpsa_motion_neighborhood_radius", 2)),
            boundary_window_seconds=float(getattr(config, "tpsa_boundary_window_seconds", 1.0)),
            boundary_nms_seconds=float(getattr(config, "tpsa_boundary_nms_seconds", 4.0)),
            boundary_expansion_seconds=float(getattr(config, "tpsa_boundary_expansion_seconds", 1.0)),
            maximum_boundary_bands=int(getattr(config, "tpsa_maximum_boundary_bands", 4)),
            query_core_fraction=float(getattr(config, "tpsa_query_core_fraction", 0.80)),
            motion_bonus_fraction=float(getattr(config, "tpsa_motion_bonus_fraction", 0.10)),
        )
        self.tpsa_allocator = TimelinePreservingSpatialAllocator(allocator_config)
        self.tpsa_fps = float(getattr(config, "tpsa_fps", 2.0))
        self.last_tpsa_stats_batch = []

    def _merged_grid_for_sample(
        self,
        video_grid_thw: Optional[torch.LongTensor],
        batch_index: int,
        frame_count: int,
        patches_per_frame: int,
    ) -> tuple[int, int]:
        if video_grid_thw is None or video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
            raise ValueError("TPSA requires explicit video_grid_thw metadata with shape [batch,3]")
        if batch_index >= video_grid_thw.shape[0]:
            raise ValueError(
                f"TPSA grid row missing for batch index {batch_index}; got {video_grid_thw.shape[0]} rows"
            )
        temporal, height, width = [int(value.item()) for value in video_grid_thw[batch_index]]
        merge = int(self.visual.spatial_merge_size)
        if height % merge or width % merge:
            raise ValueError(
                f"TPSA vision grid {(height, width)} is not divisible by spatial merge size {merge}"
            )
        merged = height // merge, width // merge
        if merged[0] * merged[1] != patches_per_frame:
            raise ValueError(
                f"TPSA merged grid mismatch: H*W={merged[0]}*{merged[1]}="
                f"{merged[0] * merged[1]}, but P={patches_per_frame}; preserve Qwen grid metadata "
                "and do not silently alter the token budget"
            )
        if temporal != frame_count:
            raise ValueError(
                f"TPSA grid reports {temporal} frames but prompt layout contains {frame_count}; "
                "use exactly one continuous [0,duration] video component per sample"
            )
        return merged

    def _semantic_prune_video_region(
        self,
        video_hidden_states: torch.FloatTensor,
        frame_token_scores: Optional[torch.FloatTensor],
        frame_global_features: torch.FloatTensor,
        query_embed: torch.FloatTensor,
        device: torch.device,
        return_coords: bool = False,
        merged_grid_hw: Optional[tuple[int, int]] = None,
    ):
        if merged_grid_hw is None:
            raise ValueError("TPSA selection hook did not receive merged (H,W) metadata")
        temporal_patch_size = max(
            int(getattr(getattr(self.visual, "patch_embed", None), "temporal_patch_size", 1)),
            1,
        )
        tubelet_fps = effective_temporal_fps(self.tpsa_fps, temporal_patch_size)
        allocation = self.tpsa_allocator(
            video_hidden_states,
            merged_grid_hw[0],
            merged_grid_hw[1],
            query_embed,
            tubelet_fps,
            retention_ratio=self.semantic_retention_ratio,
        )
        stats = allocation.stats()
        stats["spatial_policy"] = self.tpsa_allocator.config.spatial_policy
        stats["decoded_fps"] = self.tpsa_fps
        stats["temporal_patch_size"] = temporal_patch_size
        stats["effective_temporal_fps"] = tubelet_fps
        self.last_tpsa_stats_batch.append(stats)
        self.last_tpsa_stats_batch = self.last_tpsa_stats_batch[-32:]

        keep = allocation.keep_indices.to(device)
        pruned = video_hidden_states.reshape(-1, video_hidden_states.shape[-1]).to(device)[keep]
        if not return_coords:
            return pruned, keep, None

        patches = video_hidden_states.shape[1]

        def coordinates(indices: torch.Tensor) -> torch.LongTensor:
            indices = indices.to(device=device, dtype=torch.long)
            if indices.numel() == 0:
                return torch.empty((0, 2), device=device, dtype=torch.long)
            return torch.stack((indices // patches, indices % patches), dim=1)

        boundary = torch.cat((allocation.start_indices, allocation.end_indices)).sort().values
        return pruned, keep, {
            "context": coordinates(allocation.prototype_indices),
            "object": coordinates(allocation.adaptive_indices),
            "motion": coordinates(boundary),
        }
