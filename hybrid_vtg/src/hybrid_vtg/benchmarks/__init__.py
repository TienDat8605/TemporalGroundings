"""Official test-only benchmark adapters."""

from __future__ import annotations

from typing import Any


def register_benchmarks(registry: Any) -> None:
    from .momentseeker import MomentSeekerBenchmark
    from .omtg import OMTGBenchmark
    from .qvhighlights import QVHighlightsBenchmark
    from .qvhighlights_timelens import QVHighlightsTimeLensBenchmark
    from .tacos import TACoSBenchmark

    registry.register(OMTGBenchmark.name, OMTGBenchmark)
    registry.register(TACoSBenchmark.name, TACoSBenchmark)
    registry.register(QVHighlightsBenchmark.name, QVHighlightsBenchmark)
    registry.register(QVHighlightsTimeLensBenchmark.name, QVHighlightsTimeLensBenchmark)
    registry.register(MomentSeekerBenchmark.name, MomentSeekerBenchmark)
