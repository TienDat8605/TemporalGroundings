"""QVHighlights official hidden-label test split, moment retrieval only."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import Benchmark, Sample
from ..io import write_jsonl
from .common import first_file, video_index


class QVHighlightsBenchmark(Benchmark):
    name = "qvhighlights"

    def load_test(self, root: Path) -> list[Sample]:
        annotation = first_file(
            root,
            (
                "highlight_test_release.jsonl",
                "annotations/highlight_test_release.jsonl",
                "metadata/highlight_test_release.jsonl",
            ),
        )
        videos = video_index(root)
        rows = []
        with annotation.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                record = json.loads(raw)
                video = str(record["vid"])
                path = videos.get(video)
                if path is None:
                    raise FileNotFoundError(f"missing QVHighlights video {video}")
                sample = Sample(
                    id=str(record.get("qid", f"test:{line_number}")),
                    video=video,
                    video_path=path,
                    duration=float(record["duration"]),
                    query=str(record["query"]).strip(),
                    group="qvhighlights",
                )
                sample.validate()
                rows.append(sample)
        return rows

    def evaluate(self, records):
        del records
        return None

    def export_submission(self, records, destination: Path):
        values = []
        for record in records:
            prediction = record.get("prediction", {})
            spans = prediction.get("spans", [])
            values.append(
                {
                    "qid": int(record["id"]) if str(record["id"]).isdigit() else record["id"],
                    "query": record["query"],
                    "vid": record["video"],
                    "pred_relevant_windows": [
                        [float(span["start"]), float(span["end"]), float(span["score"])] for span in spans[:10]
                    ],
                }
            )
        write_jsonl(destination, values, mode="write")
        return destination
