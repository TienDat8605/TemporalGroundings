"""Scout timeline scoring and feature loading for SGDE (Idea 3)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...contracts import Sample
from ...scout_features import (DEFAULT_MODEL, DEFAULT_MODEL_REVISION, model_slug, query_identity,
                               sample_timestamps, video_identity)


@dataclass(frozen=True)
class ScoutTimeline:
    """Query-conditioned visual relevance timeline across full video duration."""

    timestamps: np.ndarray
    raw_scores: np.ndarray
    smoothed_scores: np.ndarray
    z_scores: np.ndarray
    median: float
    mad: float
    peak_z: float
    model_id: str
    cached: bool
    provenance: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.timestamps)


def smooth_timeline(scores: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply conservative moving-average / Gaussian-like smoothing to suppress single-frame blips."""
    if len(scores) <= 2 or window_size <= 1:
        return scores.copy()
    window = min(window_size, len(scores))
    if window % 2 == 0:
        window += 1
    # Triangular / quasi-Gaussian weights
    radius = window // 2
    weights = np.array([radius + 1 - abs(i - radius) for i in range(window)], dtype=np.float32)
    weights /= weights.sum()
    padded = np.pad(scores, (radius, radius), mode="edge")
    smoothed = np.convolve(padded, weights, mode="valid")
    return smoothed.astype(np.float32)


def normalize_timeline(scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Compute robust median and MAD normalized timeline: z(t) = (s(t) - median) / (MAD + eps)."""
    if len(scores) == 0:
        return np.array([], dtype=np.float32), 0.0, 1.0
    med = float(np.median(scores))
    abs_diff = np.abs(scores - med)
    mad = float(np.median(abs_diff)) * 1.4826
    eps = 1e-6
    denominator = mad if mad > 1e-5 else (float(np.std(scores)) + eps)
    z = (scores - med) / (denominator + eps)
    return z.astype(np.float32), med, denominator


class ScoutProvider:
    """Retrieves or computes scout timeline embeddings for SGDE."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        revision: str = DEFAULT_MODEL_REVISION,
        fps: float = 1.0,
        feature_roots: Sequence[Path] = (),
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.fps = fps
        self.feature_roots = tuple(Path(root) for root in feature_roots)
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _discover_feature_dirs(self, cache_root: Path, benchmark: str = "") -> list[Path]:
        """Search for feature directories containing precomputed scout embeddings."""
        candidates = []
        slug = model_slug(self.model_id)
        for root in self.feature_roots:
            if root.is_dir():
                if (root / slug).is_dir():
                    candidates.append(root / slug)
                if benchmark and (root / slug / benchmark).is_dir():
                    candidates.append(root / slug / benchmark)
                if root.name == slug:
                    candidates.append(root)
                    if benchmark and (root / benchmark).is_dir():
                        candidates.append(root / benchmark)

        # Dynamically locate project root containing assets/ or pyproject.toml
        project_root = None
        for parent in Path(__file__).resolve().parents:
            if (parent / "assets").is_dir() or (parent / "pyproject.toml").is_file():
                project_root = parent
                break
        if project_root is None:
            project_root = Path.cwd()

        assets_scout = project_root / "assets" / "features" / "scouts"
        if assets_scout.is_dir():
            candidates.append(assets_scout / slug)
            if benchmark:
                candidates.append(assets_scout / slug / benchmark)

        # Cache location
        candidates.append(cache_root / "scouts" / slug)
        if benchmark:
            candidates.append(cache_root / "scouts" / slug / benchmark)

        # Deduplicate preserving order
        unique: list[Path] = []
        seen: set[Path] = set()
        for p in candidates:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)
        return unique

    def _manifest(self, directory: Path, benchmark: str) -> dict[str, Any] | None:
        """Accept only current artifacts made by this exact scout configuration."""
        path = directory / "manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if (
            value.get("schema_version") != 2
            or value.get("model") != self.model_id
            or value.get("model_revision") != self.revision
            or value.get("benchmark") != benchmark
            or float(value.get("fps", 0.0)) != float(self.fps)
            or value.get("embedding_dtype") != "float16"
            or not isinstance(value.get("embedding_dimension"), int)
        ):
            return None
        return value

    def _validated_feature_dirs(self, cache_root: Path, benchmark: str) -> list[tuple[Path, dict[str, Any]]]:
        return [(directory, manifest) for directory in self._discover_feature_dirs(cache_root, benchmark)
                if (manifest := self._manifest(directory, benchmark)) is not None]

    def _runtime_feature_dir(self, cache_root: Path, benchmark: str) -> Path:
        return cache_root / "scouts" / model_slug(self.model_id) / benchmark

    def _write_runtime_manifest(self, directory: Path, benchmark: str, dimension: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        payload = {
            "schema_version": 2, "model": self.model_id, "model_revision": self.revision,
            "benchmark": benchmark, "fps": self.fps, "embedding_dimension": dimension,
            "embedding_dtype": "float16", "normalized": True,
            "frame_policy": "center of each fixed-rate temporal cell", "runtime_cache": True,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _load_precomputed_video_embedding(
        self,
        video_id: str,
        expected_video_identity: str,
        cache_root: Path,
        benchmark: str = "",
    ) -> tuple[np.ndarray, np.ndarray, Path, dict[str, Any]] | None:
        """Attempt to load cached video embeddings (timestamps, embeddings, model_id)."""
        for directory, manifest in self._validated_feature_dirs(cache_root, benchmark):
            # Check direct or video_embeddings subfolder
            for subpath in (
                directory / f"{video_id}.npz",
                directory / "video_embeddings" / f"{video_id}.npz",
            ):
                if subpath.is_file():
                    try:
                        with np.load(subpath, allow_pickle=False) as cached:
                            ts = cached["timestamps"].astype(np.float32)
                            emb = cached["embeddings"].astype(np.float32)
                            artifact_identity = str(cached["video_identity"].item())
                            if (ts.ndim == 1 and emb.ndim == 2 and len(ts) == len(emb)
                                    and emb.shape[1] == manifest["embedding_dimension"]
                                    and np.isfinite(emb).all() and artifact_identity == expected_video_identity):
                                return ts, emb, directory, manifest
                    except Exception:
                        continue
        return None

    def _load_precomputed_query_embedding(
        self,
        sample_id: str,
        query: str,
        cache_root: Path,
        benchmark: str = "",
        directories: Sequence[tuple[Path, dict[str, Any]]] | None = None,
    ) -> np.ndarray | None:
        """Attempt to load cached query embedding."""
        search_dirs = directories or self._validated_feature_dirs(cache_root, benchmark)
        for directory, manifest in search_dirs:
            query_path = directory / "queries.npz"
            if query_path.is_file():
                try:
                    with np.load(query_path, allow_pickle=False) as cached:
                        ids = [str(val) for val in cached["ids"]]
                        embeddings = cached["embeddings"].astype(np.float32)
                        identities = [str(value) for value in cached["query_identities"]]
                        if (embeddings.ndim == 2 and embeddings.shape[1] == manifest["embedding_dimension"]
                                and np.isfinite(embeddings).all() and sample_id in ids):
                            idx = ids.index(sample_id)
                            if identities[idx] == query_identity(query):
                                return embeddings[idx]
                except Exception:
                    pass
        return None

    def _get_model(self) -> Any:
        if self._model is None:
            from ...scout_features import _load_model

            self._model = _load_model(self.model_id, self.revision, self.device)
            if hasattr(self._model, "processor"):
                self._model.processor.p_max_length = 2048
                self._model.processor.max_input_tiles = 1
                self._model.processor.use_thumbnail = True
        return self._model

    def unload(self) -> None:
        """Release scout model GPU memory."""
        import gc

        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def compute_timeline(
        self,
        sample: Sample,
        cache_root: Path,
        *,
        smoothing_window: int = 3,
    ) -> ScoutTimeline:
        """Compute normalized scout timeline z(t) for one sample."""
        benchmark = sample.group
        cached_vid = self._load_precomputed_video_embedding(
            sample.video, video_identity(sample.video_path), cache_root, benchmark
        )
        cached_q = None
        if cached_vid is not None:
            cached_q = self._load_precomputed_query_embedding(
                sample.id, sample.query, cache_root, benchmark, directories=[(cached_vid[2], cached_vid[3])]
            )

        timestamps: np.ndarray
        video_emb: np.ndarray
        query_emb: np.ndarray
        was_cached = False
        provenance: dict[str, Any] = {"model": self.model_id, "revision": self.revision, "fps": self.fps}

        if cached_vid is not None:
            timestamps, video_emb, directory, manifest = cached_vid
            provenance = {**manifest, "feature_dir": str(directory)}
        else:
            # Compute on the fly and cache
            timestamps = sample_timestamps(sample.duration, self.fps)
            model = self._get_model()
            from ...scout_features import _encode_video

            video_emb = _encode_video(model, sample.video_path, timestamps, batch_size=1).astype(np.float32)
            # Save to cache
            feature_dir = self._runtime_feature_dir(cache_root, benchmark)
            dest = feature_dir / "video_embeddings" / f"{sample.video}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dest, timestamps=timestamps, embeddings=video_emb.astype(np.float16),
                video_identity=np.asarray(video_identity(sample.video_path)),
            )
            self._write_runtime_manifest(feature_dir, benchmark, int(video_emb.shape[1]))

        if cached_q is not None:
            query_emb = cached_q
            was_cached = cached_vid is not None
        else:
            model = self._get_model()
            from ...scout_features import _encode_queries

            query_emb = _encode_queries(model, [sample.query], batch_size=1)[0].astype(np.float32)
            # Save to cache
            feature_dir = self._runtime_feature_dir(cache_root, benchmark)
            dest = feature_dir / "queries.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file():
                try:
                    with np.load(dest, allow_pickle=False) as existing:
                        ids = list(existing["ids"])
                        embs = list(existing["embeddings"])
                        identities = list(existing["query_identities"])
                        if sample.id not in ids:
                            ids.append(sample.id)
                            embs.append(query_emb.astype(np.float16))
                            identities.append(query_identity(sample.query))
                            np.savez_compressed(dest, ids=np.asarray(ids), embeddings=np.asarray(embs), query_identities=np.asarray(identities))
                except Exception:
                    np.savez_compressed(
                        dest, ids=np.asarray([sample.id]), embeddings=np.asarray([query_emb.astype(np.float16)]), query_identities=np.asarray([query_identity(sample.query)])
                    )
            else:
                np.savez_compressed(
                    dest, ids=np.asarray([sample.id]), embeddings=np.asarray([query_emb.astype(np.float16)]), query_identities=np.asarray([query_identity(sample.query)])
                )
            self._write_runtime_manifest(feature_dir, benchmark, int(query_emb.shape[0]))

        # Normalize rows
        v_norms = np.linalg.norm(video_emb, axis=1, keepdims=True)
        v_normed = video_emb / np.maximum(v_norms, 1e-8)
        q_norm = query_emb / max(float(np.linalg.norm(query_emb)), 1e-8)
        if video_emb.shape[1] != query_emb.shape[0]:
            raise ValueError("scout video/query embeddings have mismatched provenance dimensions")

        raw_scores = np.dot(v_normed, q_norm).astype(np.float32)
        smoothed_scores = smooth_timeline(raw_scores, smoothing_window)
        z_scores, med, mad = normalize_timeline(smoothed_scores)

        peak_z = float(np.max(z_scores)) if len(z_scores) > 0 else 0.0

        return ScoutTimeline(
            timestamps=timestamps,
            raw_scores=raw_scores,
            smoothed_scores=smoothed_scores,
            z_scores=z_scores,
            median=med,
            mad=mad,
            peak_z=peak_z,
            model_id=self.model_id,
            cached=was_cached,
            provenance=provenance,
        )
