"""Frozen SigLIP encoder used for cheap global search and local refinement."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np
from PIL import Image


def normalize_rows(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


class FrozenSiglipEncoder:
    """Thin inference-only adapter; imports model dependencies only when instantiated."""

    def __init__(self, checkpoint: str, *, batch_size: int = 32, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("torch and transformers are required for coarse encoding") from error
        self._torch = torch
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        # Keep the checkpoint's current image preprocessing behavior explicit. This avoids
        # Transformers silently switching to the fast processor in a future release.
        self.processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False)
        self.model = AutoModel.from_pretrained(checkpoint, dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)
        self.last_encode_stats: dict[str, float | int] = {}

    def _move(self, values: dict) -> dict:
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in values.items()}

    def encode_text(self, text: str) -> np.ndarray:
        values = self.processor(text=[text], padding=True, truncation=True, return_tensors="pt")
        with self._torch.inference_mode():
            output = self.model.get_text_features(**self._move(values))
        return normalize_rows(output.float().cpu().numpy())[0]

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("at least one image is required")
        chunks = []
        processor_seconds = 0.0
        vision_seconds = 0.0
        input_pixels = 0
        for start in range(0, len(images), self.batch_size):
            processor_started = perf_counter()
            values = self.processor(images=list(images[start:start + self.batch_size]), return_tensors="pt")
            processor_seconds += perf_counter() - processor_started
            pixel_values = values.get("pixel_values")
            if pixel_values is not None and getattr(pixel_values, "ndim", 0) >= 4:
                input_pixels += int(pixel_values.shape[0] * pixel_values.shape[-2] * pixel_values.shape[-1])
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
            vision_started = perf_counter()
            with self._torch.inference_mode():
                output = self.model.get_image_features(**self._move(values))
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
            vision_seconds += perf_counter() - vision_started
            chunks.append(output.float().cpu().numpy())
        self.last_encode_stats = {
            "frames": len(images),
            "input_pixels": input_pixels,
            "processor_seconds": processor_seconds,
            "vision_encoder_seconds": vision_seconds,
        }
        return normalize_rows(np.concatenate(chunks, axis=0))
