"""Precompute reusable image/text scout embeddings for temporal grounding."""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .downloads import VIDEO_SUFFIXES
from .io import write_json

DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEFAULT_MODEL_REVISION = "582e3bf72aee355e3c59ed89de53543c5b0657ee"
DEFAULT_BENCHMARK = "qvhighlights-timelens"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def sample_timestamps(duration: float, fps: float) -> np.ndarray:
    """Return centered timestamps for fixed-rate temporal cells."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be positive and finite, got {duration}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    count = max(1, math.ceil(duration * fps))
    values = (np.arange(count, dtype=np.float64) + 0.5) / fps
    return np.minimum(values, np.nextafter(duration, 0.0)).astype(np.float32)


def annotation_queries(annotation: dict[str, Any]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    queries: list[str] = []
    for video_id, record in annotation.items():
        values = record.get("queries")
        if not isinstance(values, list) or not values:
            raise ValueError(f"annotation has no queries for {video_id}")
        for index, query in enumerate(values):
            query = str(query).strip()
            if not query:
                raise ValueError(f"annotation has an empty query for {video_id}::{index}")
            ids.append(f"{video_id}::{index}")
            queries.append(query)
    return ids, queries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_identity(path: Path) -> str:
    """Cheap revision identity for cache invalidation without decoding the video."""
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def query_identity(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _normalized_float16(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"embeddings must be rank two, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("embeddings contain a non-finite or zero-norm row")
    return (array / norms).astype(np.float16)


def _batches(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _video_index(root: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.stem in videos:
            raise ValueError(f"duplicate video stem {path.stem}: {videos[path.stem]} and {path}")
        videos[path.stem] = path
    return videos


def _frame_batches(
    video_path: Path,
    timestamps: Sequence[float],
    batch_size: int,
) -> Iterator[list[Image.Image]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        for timestamp_batch in _batches(timestamps, batch_size):
            frames: list[Image.Image] = []
            for timestamp in timestamp_batch:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                        ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"cannot decode {video_path} at {timestamp:.3f}s")
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            yield frames
    finally:
        capture.release()


def _load_model(model_id: str, revision: str, device: str) -> Any:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.init()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map: str | dict[str, str] = {"": device}
    else:
        dtype = torch.float32
        device_map = {"": device}
    checkpoint = snapshot_download(repo_id=model_id, revision=revision)
    if model_id.startswith("google/siglip2-"):
        from transformers import AutoProcessor

        model = AutoModel.from_pretrained(
            checkpoint, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
        ).eval()
        return _SigLIP2Adapter(model, AutoProcessor.from_pretrained(checkpoint), device)
    return AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()


class _SigLIP2Adapter:
    """Give standard Transformers SigLIP2 the scout encoder interface."""

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def _inputs(self, values: Any) -> dict[str, Any]:
        return {name: value.to(self.device) if hasattr(value, "to") else value for name, value in values.items()}

    def encode_queries(self, queries: Sequence[str]) -> Any:
        values = self._inputs(self.processor(text=list(queries), padding=True, return_tensors="pt"))
        return self.model.get_text_features(**values)

    def encode_documents(self, *, images: Sequence[Image.Image]) -> Any:
        values = self._inputs(self.processor(images=list(images), return_tensors="pt"))
        return self.model.get_image_features(**values)


def _encode_queries(model: Any, queries: Sequence[str], batch_size: int) -> np.ndarray:
    import torch

    encoded = []
    with torch.inference_mode():
        for batch in tqdm(tuple(_batches(queries, batch_size)), desc="query embeddings", unit="batch"):
            encoded.append(_normalized_float16(model.encode_queries(list(batch))))
    return np.concatenate(encoded, axis=0)


def _encode_video(
    model: Any,
    video_path: Path,
    timestamps: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    import torch

    encoded = []
    with torch.inference_mode():
        for frames in _frame_batches(video_path, timestamps.tolist(), batch_size):
            encoded.append(_normalized_float16(model.encode_documents(images=frames)))
    return np.concatenate(encoded, axis=0)


def _valid_video_cache(path: Path, timestamps: np.ndarray, dimension: int, identity: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as value:
            cached_timestamps = value["timestamps"]
            embeddings = value["embeddings"]
            return (
                cached_timestamps.dtype == np.float32
                and embeddings.dtype == np.float16
                and embeddings.shape == (len(timestamps), dimension)
                and np.array_equal(cached_timestamps, timestamps)
                and np.isfinite(embeddings).all()
                and str(value["video_identity"].item()) == identity
            )
    except (KeyError, OSError, ValueError):
        return False


def create_archive(feature_dir: Path) -> Path:
    archive = feature_dir.parent.parent / f"scout_{feature_dir.parent.name}_{feature_dir.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(feature_dir, arcname=f"{feature_dir.parent.name}/{feature_dir.name}")
    return archive


def extract_scout_features(
    dataset_root: Path,
    output_root: Path,
    *,
    model_id: str = DEFAULT_MODEL,
    revision: str = DEFAULT_MODEL_REVISION,
    benchmark: str = DEFAULT_BENCHMARK,
    fps: float = 1.0,
    batch_size: int = 1,
    query_batch_size: int = 4,
    max_input_tiles: int = 1,
    device: str = "cuda:0",
    limit_videos: int | None = None,
    archive: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    annotation_path = dataset_root / f"{benchmark}.json"
    videos_root = dataset_root / "videos"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"missing annotation: {annotation_path}")
    if not videos_root.is_dir():
        raise FileNotFoundError(f"missing videos: {videos_root}")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    query_ids, queries = annotation_queries(annotation)
    videos = _video_index(videos_root)
    missing = sorted(set(annotation) - set(videos))
    if missing:
        raise FileNotFoundError(f"dataset is missing {len(missing)} videos: {missing[:5]}")

    feature_dir = output_root.expanduser().resolve() / model_slug(model_id) / benchmark
    video_output = feature_dir / "video_embeddings"
    feature_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_id, revision, device)
    if hasattr(model.processor, "p_max_length"):
        model.processor.p_max_length = 2048
        model.processor.max_input_tiles = max_input_tiles
        model.processor.use_thumbnail = True

    query_path = feature_dir / "queries.npz"
    query_is_current = False
    if query_path.is_file():
        try:
            with np.load(query_path, allow_pickle=False) as value:
                query_is_current = (
                    np.array_equal(value["ids"], np.asarray(query_ids))
                    and np.array_equal(value["query_identities"], np.asarray([query_identity(query) for query in queries]))
                    and value["embeddings"].ndim == 2
                    and np.isfinite(value["embeddings"]).all()
                )
        except (KeyError, OSError, ValueError):
            query_is_current = False
    if not query_is_current:
        query_embeddings = _encode_queries(model, queries, query_batch_size)
        _atomic_npz(
            query_path,
            ids=np.asarray(query_ids),
            embeddings=query_embeddings,
            query_identities=np.asarray([query_identity(query) for query in queries]),
        )
    with np.load(query_path, allow_pickle=False) as query_cache:
        if not np.array_equal(query_cache["ids"], np.asarray(query_ids)):
            raise ValueError(f"query cache IDs do not match annotation: {query_path}")
        dimension = int(query_cache["embeddings"].shape[1])

    selected_ids = list(annotation)
    if limit_videos is not None:
        selected_ids = selected_ids[:limit_videos]
    completed = 0
    for video_id in tqdm(selected_ids, desc="video embeddings", unit="video"):
        duration = float(annotation[video_id]["duration"])
        timestamps = sample_timestamps(duration, fps)
        destination = video_output / f"{video_id}.npz"
        if _valid_video_cache(destination, timestamps, dimension, video_identity(videos[video_id])):
            completed += 1
            continue
        embeddings = _encode_video(model, videos[video_id], timestamps, batch_size)
        if embeddings.shape != (len(timestamps), dimension):
            raise ValueError(
                f"unexpected embedding shape for {video_id}: {embeddings.shape}, "
                f"expected {(len(timestamps), dimension)}"
            )
        _atomic_npz(
            destination, timestamps=timestamps, embeddings=embeddings,
            video_identity=np.asarray(video_identity(videos[video_id])),
        )
        completed += 1

    manifest = {
        "schema_version": 2,
        "model": model_id,
        "model_revision": revision,
        "benchmark": benchmark,
        "dataset_annotation": str(annotation_path),
        "dataset_annotation_sha256": _sha256(annotation_path),
        "fps": fps,
        "embedding_dimension": dimension,
        "embedding_dtype": "float16",
        "normalized": True,
        "frame_policy": "center of each fixed-rate temporal cell",
        "max_input_tiles": max_input_tiles,
        "use_thumbnail": True,
        "query_count": len(query_ids),
        "expected_video_count": len(annotation),
        "completed_video_count": completed,
        "complete": completed == len(annotation),
    }
    write_json(feature_dir / "manifest.json", manifest)
    if archive and manifest["complete"]:
        manifest["archive"] = str(create_archive(feature_dir))
        write_json(feature_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_REVISION",
    "annotation_queries",
    "create_archive",
    "extract_scout_features",
    "model_slug",
    "query_identity",
    "sample_timestamps",
    "video_identity",
]
