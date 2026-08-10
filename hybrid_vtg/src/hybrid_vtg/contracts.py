"""Small, model-agnostic contracts shared by methods and benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence


@dataclass(frozen=True)
class Sample:
    id: str
    video: str
    video_path: Path
    duration: float
    query: str
    targets: tuple[tuple[float, float], ...] = ()
    group: str = ""
    cardinality: str = "single"

    def validate(self) -> None:
        if not self.id or not self.query.strip():
            raise ValueError("sample id and query are required")
        if not self.video_path.is_file():
            raise FileNotFoundError(f"video does not exist: {self.video_path}")
        if self.duration <= 0:
            raise ValueError(f"sample {self.id!r} has a non-positive duration")
        if self.cardinality not in {"single", "multi"}:
            raise ValueError(f"invalid cardinality: {self.cardinality}")
        for start, end in self.targets:
            if not 0 <= start < end <= self.duration + 1e-3:
                raise ValueError(f"sample {self.id!r} has invalid target {(start, end)}")


@dataclass(frozen=True)
class ScoredSpan:
    start: float
    end: float
    score: float = 1.0

    def clipped(self, duration: float) -> "ScoredSpan | None":
        start, end = max(0.0, self.start), min(duration, self.end)
        return ScoredSpan(start, end, self.score) if end > start else None


@dataclass(frozen=True)
class Prediction:
    spans: tuple[ScoredSpan, ...]
    raw_output: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": [asdict(span) for span in self.spans],
            "raw_output": self.raw_output,
            "telemetry": dict(self.telemetry),
        }


@dataclass
class TemporalEvidence:
    """Encoded visual units with one absolute timestamp per embedding row."""

    embeddings: Any
    timestamps: tuple[float, ...]
    source_frames: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rows = int(self.embeddings.shape[0])
        if rows != len(self.timestamps):
            raise ValueError(f"evidence has {rows} rows but {len(self.timestamps)} timestamps")
        if rows == 0:
            raise ValueError("evidence cannot be empty")

    @property
    def size(self) -> int:
        return len(self.timestamps)

    def select(self, indices: Sequence[int]) -> "TemporalEvidence":
        import torch

        index = torch.as_tensor(indices, device=self.embeddings.device, dtype=torch.long)
        return TemporalEvidence(
            embeddings=self.embeddings.index_select(0, index),
            timestamps=tuple(self.timestamps[int(value)] for value in index.cpu().tolist()),
            source_frames=self.source_frames,
            metadata={**self.metadata, "selected_indices": index.cpu().tolist()},
        )

    @classmethod
    def concatenate(cls, values: Sequence["TemporalEvidence"]) -> "TemporalEvidence":
        import torch

        if not values:
            raise ValueError("at least one evidence block is required")
        embeddings = torch.cat([value.embeddings for value in values], dim=0)
        timestamps = tuple(time for value in values for time in value.timestamps)
        order = sorted(range(len(timestamps)), key=lambda index: (timestamps[index], index))
        merged = cls(
            embeddings=embeddings,
            timestamps=timestamps,
            source_frames=sum(value.source_frames for value in values),
            metadata={"parts": [value.metadata for value in values]},
        )
        return merged.select(order)


@dataclass(frozen=True)
class GroundingContext:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class ModelBackend(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[frozenset[str]] = frozenset({"encoded-evidence"})

    @abstractmethod
    def encode(self, sample: Sample, timestamps: Sequence[float]) -> TemporalEvidence:
        """Encode frames sampled at absolute source-video timestamps."""

    @abstractmethod
    def query_scores(self, evidence: TemporalEvidence, query: str) -> Any:
        """Return one query-relevance score per evidence row."""

    @abstractmethod
    def predict(
        self,
        sample: Sample,
        evidence: TemporalEvidence,
        context: GroundingContext,
    ) -> Prediction:
        """Predict global timestamped spans from encoded evidence."""

    @property
    def maximum_evidence_units(self) -> int | None:
        return None


class Method(ABC):
    name: ClassVar[str]
    required_capabilities: ClassVar[frozenset[str]] = frozenset({"encoded-evidence"})

    def validate_model(self, model: ModelBackend) -> None:
        missing = self.required_capabilities - model.capabilities
        if missing:
            raise ValueError(
                f"method {self.name!r} requires {sorted(missing)}; "
                f"model {model.name!r} provides {sorted(model.capabilities)}"
            )

    @abstractmethod
    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        """Ground one sample."""


class Benchmark(ABC):
    name: ClassVar[str]

    @abstractmethod
    def load_test(self, root: Path) -> list[Sample]:
        """Load the official test split."""

    @abstractmethod
    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        """Evaluate successful records, or return None when labels are hidden."""

    def export_submission(
        self,
        records: Sequence[Mapping[str, Any]],
        destination: Path,
    ) -> Path | None:
        return None
