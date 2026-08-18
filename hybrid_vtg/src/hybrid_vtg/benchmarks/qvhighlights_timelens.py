"""TencentARC/TimeLens-Bench QVHighlights split adapter."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import Benchmark, Sample
from ..metrics import evaluate_records
from .common import first_file, video_index


class QVHighlightsTimeLensBenchmark(Benchmark):
    name = "qvhighlights-timelens"

    def load_test(self, root: Path) -> list[Sample]:
        annotation = first_file(
            root,
            (
                "qvhighlights-timelens.json",
                "annotations/qvhighlights-timelens.json",
                "metadata/qvhighlights-timelens.json",
            ),
        )
        videos = video_index(root)
        with annotation.open(encoding="utf-8") as handle:
            data = json.load(handle)

        rows = []
        for vid, record in sorted(data.items()):
            path = videos.get(vid)
            if path is None:
                raise FileNotFoundError(f"missing QVHighlights video {vid}")
            duration = float(record.get("duration") or 0.0)
            if duration <= 0:
                from ..media import probe_video

                duration = probe_video(path).duration
            spans = record.get("spans", [])
            queries = record.get("queries", [])
            for query_index, query in enumerate(queries):
                span = spans[query_index] if query_index < len(spans) else None
                targets = ()
                if span and len(span) >= 2 and float(span[1]) > float(span[0]):
                    targets = ((max(0.0, float(span[0])), min(duration, float(span[1]))),)
                sample = Sample(
                    id=f"{vid}::{query_index}",
                    video=vid,
                    video_path=path,
                    duration=duration,
                    query=str(query).strip(),
                    targets=targets,
                    group="qvhighlights-timelens",
                    cardinality="single",
                )
                sample.validate()
                rows.append(sample)
        return rows

    def evaluate(self, records):
        return evaluate_records(records, multi_span=False)


__all__ = ["QVHighlightsTimeLensBenchmark"]
