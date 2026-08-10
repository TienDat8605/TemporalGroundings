"""Extensible, training-free video temporal grounding benchmarks."""

from .contracts import Benchmark, Method, ModelBackend, Prediction, Sample, ScoredSpan

__all__ = ["Benchmark", "Method", "ModelBackend", "Prediction", "Sample", "ScoredSpan"]
__version__ = "0.2.0"
