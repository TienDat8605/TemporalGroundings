"""Raw and pre-extracted feature providers for official UniVTG feature stacks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from ...contracts import Sample
from ...media import extract_frames, uniform_timestamps


class UniVTGFeatures:
    def __init__(self, cache_dir: Path, feature_stack: str, feature_roots: tuple[Path, ...] = ()) -> None:
        self.cache_dir = cache_dir
        self.feature_stack = feature_stack
        self.feature_roots = tuple(path.expanduser().resolve() for path in feature_roots)
        self._clip_model = None
        self._clip_processor = None
        self._slowfast = None

    @property
    def clip_checkpoint(self) -> str:
        return "openai/clip-vit-base-patch16" if "b16" in self.feature_stack else "openai/clip-vit-base-patch32"

    def _key(self, sample: Sample, timestamps: Sequence[float]) -> str:
        payload = f"{sample.video_path.resolve()}\0{self.feature_stack}\0" + ",".join(
            f"{value:.6f}" for value in timestamps
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _precomputed(self, sample: Sample, timestamps: Sequence[float]) -> np.ndarray | None:
        if not self.feature_roots:
            return None
        arrays = []
        for root in self.feature_roots:
            path = root / f"{Path(sample.video).stem}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing UniVTG feature file: {path}")
            with np.load(path) as archive:
                arrays.append(archive["features"].astype(np.float32))
        length = min(len(array) for array in arrays)
        combined = np.concatenate([array[:length] for array in arrays], axis=-1)
        if length == 0:
            raise ValueError(f"empty UniVTG features for {sample.video}")
        positions = np.asarray(timestamps, dtype=np.float64) / max(sample.duration, 1e-6)
        indices = np.rint(positions.clip(0, 1) * (length - 1)).astype(np.int64)
        return combined[indices]

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_checkpoint)
            self._clip_model = CLIPModel.from_pretrained(self.clip_checkpoint).eval()
            if __import__("torch").cuda.is_available():
                self._clip_model.cuda()
        return self._clip_model, self._clip_processor

    def _clip_images(self, paths: Sequence[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        model, processor = self._load_clip()
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            values = model.get_image_features(**inputs)
        return torch.nn.functional.normalize(values.float(), dim=-1).cpu().numpy()

    def text(self, query: str):
        import torch

        model, processor = self._load_clip()
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.text_model(**inputs).last_hidden_state[0]
        length = int(inputs["attention_mask"][0].sum())
        return output[:length].float()

    def _load_slowfast(self):
        if self._slowfast is None:
            try:
                from pytorchvideo.models.hub import slowfast_r50
            except ImportError as error:
                raise RuntimeError(
                    "SlowFast UniVTG checkpoints require pytorchvideo or official pre-extracted features"
                ) from error
            model = slowfast_r50(pretrained=True).eval()
            model.blocks[-1].proj = __import__("torch").nn.Identity()
            if __import__("torch").cuda.is_available():
                model.cuda()
            self._slowfast = model
        return self._slowfast

    def _slowfast_images(self, sample: Sample, timestamps: Sequence[float]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        model = self._load_slowfast()
        rows = []
        for index, center in enumerate(timestamps):
            clip_times = uniform_timestamps(max(0.0, center - 1.0), min(sample.duration, center + 1.0), 32)
            paths = extract_frames(
                sample.video_path, clip_times, self.cache_dir / f"slowfast-{index}", maximum_width=256
            )
            frames = []
            for path in paths:
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                frames.append(torch.from_numpy(array).permute(2, 0, 1))
            fast = torch.stack(frames, dim=1).unsqueeze(0)
            fast = functional.interpolate(fast, size=(32, 256, 256), mode="trilinear", align_corners=False)
            mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)
            fast = (fast - mean) / std
            fast = fast.to(next(model.parameters()).device)
            slow = fast[:, :, ::4]
            with torch.inference_mode():
                value = model([slow, fast]).reshape(1, -1)
            rows.append(value[0, :2304].float().cpu())
        return torch.stack(rows).numpy()

    def video(self, sample: Sample, timestamps: Sequence[float]) -> np.ndarray:
        precomputed = self._precomputed(sample, timestamps)
        if precomputed is not None:
            return precomputed
        key = self._key(sample, timestamps)
        path = self.cache_dir / "univtg" / self.feature_stack / f"{key}.npz"
        if path.is_file():
            return np.load(path)["features"].astype(np.float32)
        paths = extract_frames(sample.video_path, timestamps, self.cache_dir / "frames", maximum_width=336)
        clip = self._clip_images(paths)
        values = (
            np.concatenate([self._slowfast_images(sample, timestamps), clip], axis=-1)
            if self.feature_stack.startswith("slowfast")
            else clip
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=values.astype(np.float16))
        return values.astype(np.float32)
