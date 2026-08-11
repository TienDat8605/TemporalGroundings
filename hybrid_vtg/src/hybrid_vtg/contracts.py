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
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rows = int(self.embeddings.shape[0])
        if rows != len(self.timestamps):
            raise ValueError(f"evidence has {rows} rows but {len(self.timestamps)} timestamps")
        if rows == 0:
            raise ValueError("evidence cannot be empty")
        if self.roles and len(self.roles) != rows:
            raise ValueError(f"evidence has {rows} rows but {len(self.roles)} roles")

    @property
    def size(self) -> int:
        return len(self.timestamps)

    def select(self, indices: Sequence[int]) -> "TemporalEvidence":
        import torch

        index = torch.as_tensor(indices, device=self.embeddings.device, dtype=torch.long)
        selected = index.cpu().tolist()
        metadata = {**self.metadata, "selected_indices": selected}
        coordinates = self.metadata.get("cell_coordinates")
        if isinstance(coordinates, list) and len(coordinates) == self.size:
            metadata["cell_coordinates"] = [coordinates[int(value)] for value in selected]
        return TemporalEvidence(
            embeddings=self.embeddings.index_select(0, index),
            timestamps=tuple(self.timestamps[int(value)] for value in selected),
            source_frames=self.source_frames,
            metadata=metadata,
            roles=tuple(self.roles[int(value)] for value in selected) if self.roles else (),
        )

    @classmethod
    def concatenate(cls, values: Sequence["TemporalEvidence"]) -> "TemporalEvidence":
        import torch

        if not values:
            raise ValueError("at least one evidence block is required")
        embeddings = torch.cat([value.embeddings for value in values], dim=0)
        timestamps = tuple(time for value in values for time in value.timestamps)
        roles = tuple(role for value in values for role in value.roles) if all(value.roles for value in values) else ()
        order = sorted(range(len(timestamps)), key=lambda index: (timestamps[index], index))
        coordinates = []
        for value in values:
            part = value.metadata.get("cell_coordinates")
            if not isinstance(part, list) or len(part) != value.size:
                coordinates = []
                break
            coordinates.extend(part)
        metadata = {"parts": [value.metadata for value in values]}
        if coordinates:
            metadata["cell_coordinates"] = coordinates
        merged = cls(
            embeddings=embeddings,
            timestamps=timestamps,
            source_frames=sum(value.source_frames for value in values),
            metadata=metadata,
            roles=roles,
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

    def prepare(self, samples: Sequence[Sample], cache_root: Path) -> None:
        """Run one batch-level pass over all pending samples before grounding.

        Called by the runner once, before the model backend is loaded, with the full
        set of pending samples. The default is a no-op; methods that benefit from a
        batch CPU-only preprocessing step (e.g. scene detection) override it. The model
        is not loaded yet, so this must not require GPU memory.
        """


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
