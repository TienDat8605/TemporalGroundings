"""Scout timeline scoring and feature loading for SGDE (Idea 3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...contracts import Sample
from ...scout_features import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, model_slug, sample_timestamps


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
                candidates.append(root)
                if (root / slug).is_dir():
                    candidates.append(root / slug)
                if benchmark and (root / slug / benchmark).is_dir():
                    candidates.append(root / slug / benchmark)
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
            for sub in assets_scout.iterdir():
                if sub.is_dir():
                    candidates.append(sub)
                    if benchmark and (sub / benchmark).is_dir():
                        candidates.append(sub / benchmark)

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

    def _load_precomputed_video_embedding(
        self,
        video_id: str,
        cache_root: Path,
        benchmark: str = "",
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Attempt to load cached video embeddings (timestamps, embeddings, model_id)."""
        search_dirs = self._discover_feature_dirs(cache_root, benchmark)
        for directory in search_dirs:
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
                            if ts.ndim == 1 and emb.ndim == 2 and len(ts) == len(emb):
                                model_name = directory.parent.name if directory.name == benchmark else directory.name
                                return ts, emb, model_name
                    except Exception:
                        continue
        return None

    def _load_precomputed_query_embedding(
        self,
        sample_id: str,
        query: str,
        cache_root: Path,
        benchmark: str = "",
    ) -> np.ndarray | None:
        """Attempt to load cached query embedding."""
        search_dirs = self._discover_feature_dirs(cache_root, benchmark)
        for directory in search_dirs:
            query_path = directory / "queries.npz"
            if query_path.is_file():
                try:
                    with np.load(query_path, allow_pickle=False) as cached:
                        ids = [str(val) for val in cached["ids"]]
                        embeddings = cached["embeddings"].astype(np.float32)
                        if sample_id in ids:
                            idx = ids.index(sample_id)
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
        cached_vid = self._load_precomputed_video_embedding(sample.video, cache_root, benchmark)
        cached_q = self._load_precomputed_query_embedding(sample.id, sample.query, cache_root, benchmark)

        timestamps: np.ndarray
        video_emb: np.ndarray
        query_emb: np.ndarray
        was_cached = False
        detected_model = self.model_id

        if cached_vid is not None:
            timestamps, video_emb, detected_model = cached_vid
        else:
            # Compute on the fly and cache
            timestamps = sample_timestamps(sample.duration, self.fps)
            model = self._get_model()
            from ...scout_features import _encode_video

            video_emb = _encode_video(model, sample.video_path, timestamps, batch_size=64).astype(np.float32)
            # Save to cache
            dest = (
                cache_root
                / "scouts"
                / model_slug(self.model_id)
                / benchmark
                / "video_embeddings"
                / f"{sample.video}.npz"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, timestamps=timestamps, embeddings=video_emb.astype(np.float16))

        if cached_q is not None:
            query_emb = cached_q
            was_cached = cached_vid is not None
        else:
            model = self._get_model()
            from ...scout_features import _encode_queries

            query_emb = _encode_queries(model, [sample.query], batch_size=1)[0].astype(np.float32)
            # Save to cache
            dest = cache_root / "scouts" / model_slug(self.model_id) / benchmark / "queries.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file():
                try:
                    with np.load(dest, allow_pickle=False) as existing:
                        ids = list(existing["ids"])
                        embs = list(existing["embeddings"])
                        if sample.id not in ids:
                            ids.append(sample.id)
                            embs.append(query_emb.astype(np.float16))
                            np.savez_compressed(dest, ids=np.asarray(ids), embeddings=np.asarray(embs))
                except Exception:
                    np.savez_compressed(
                        dest, ids=np.asarray([sample.id]), embeddings=np.asarray([query_emb.astype(np.float16)])
                    )
            else:
                np.savez_compressed(
                    dest, ids=np.asarray([sample.id]), embeddings=np.asarray([query_emb.astype(np.float16)])
                )

        # Normalize rows
        v_norms = np.linalg.norm(video_emb, axis=1, keepdims=True)
        v_normed = video_emb / np.maximum(v_norms, 1e-8)
        q_norm = query_emb / max(float(np.linalg.norm(query_emb)), 1e-8)

        raw_scores = np.dot(v_normed, q_norm).astype(np.float32)
        smoothed_scores = smooth_timeline(raw_scores, smoothing_window)
        z_scores, med, mad = normalize_timeline(smoothed_scores)

        peak_z = float(np.max(z_scores)) if len(z_scores) > 0 else 0.0

        return ScoutTimeline(
            timestamps=timestamps,
            raw_scores=raw_scores,
            smoothed_scores=smoothed_scores,
            z_scores=z_scores,
            peak_z=peak_z,
            median=med,
            mad=mad,
            fps=self.fps,
            model_id=detected_model,
            was_cached=was_cached,
        )

    def prepare_batch(self, samples: Sequence[Sample], cache_root: Path) -> None:
        """High-performance vectorized batch pre-extraction of queries and videos."""
        if not samples:
            return
        benchmark = samples[0].group
        scout_dir = cache_root / "scouts" / model_slug(self.model_id) / benchmark
        video_dir = scout_dir / "video_embeddings"
        video_dir.mkdir(parents=True, exist_ok=True)

        # 1. Batch encode all missing queries in parallel
        missing_ids = []
        missing_queries = []
        for s in samples:
            if self._load_precomputed_query_embedding(s.id, s.query, cache_root, benchmark) is None:
                missing_ids.append(s.id)
                missing_queries.append(s.query)

        if missing_queries:
            model = self._get_model()
            from ...scout_features import _encode_queries

            new_embs = _encode_queries(model, missing_queries, batch_size=64)
            dest = scout_dir / "queries.npz"
            all_ids = []
            all_embs = []
            if dest.is_file():
                try:
                    with np.load(dest, allow_pickle=False) as existing:
                        all_ids = [str(x) for x in existing["ids"]]
                        all_embs = list(existing["embeddings"])
                except Exception:
                    pass
            for sid, emb in zip(missing_ids, new_embs):
                if sid not in all_ids:
                    all_ids.append(sid)
                    all_embs.append(emb.astype(np.float16))
            np.savez_compressed(dest, ids=np.asarray(all_ids), embeddings=np.asarray(all_embs))

        # 2. Extract only missing unique videos
        unique_missing: dict[str, Sample] = {}
        for s in samples:
            if s.video not in unique_missing and self._load_precomputed_video_embedding(s.video, cache_root, benchmark) is None:
                unique_missing[s.video] = s

        if unique_missing:
            model = self._get_model()
            from ...scout_features import _encode_video
            from tqdm import tqdm

            device = self.device or "cuda:0"
            progress = tqdm(
                list(unique_missing.values()),
                desc=f"sgde scout video cache ({device})",
                unit="video",
            )
            for sample in progress:
                timestamps = sample_timestamps(sample.duration, self.fps)
                video_emb = _encode_video(model, sample.video_path, timestamps, batch_size=64).astype(np.float32)
                dest = video_dir / f"{sample.video}.npz"
                np.savez_compressed(dest, timestamps=timestamps, embeddings=video_emb.astype(np.float16))
