"""Frozen SigLIP encoder used for cheap global search and local refinement."""

from __future__ import annotations

from collections.abc import Sequence

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
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)

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
        for start in range(0, len(images), self.batch_size):
            values = self.processor(images=list(images[start:start + self.batch_size]), return_tensors="pt")
            with self._torch.inference_mode():
                output = self.model.get_image_features(**self._move(values))
            chunks.append(output.float().cpu().numpy())
        return normalize_rows(np.concatenate(chunks, axis=0))
