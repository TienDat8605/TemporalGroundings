"""TACoS test split adapter."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import Benchmark, Sample
from ..metrics import evaluate_records
from .common import first_file, video_index


class TACoSBenchmark(Benchmark):
    name = "tacos"

    def load_test(self, root: Path) -> list[Sample]:
        annotation = first_file(root, ("test.jsonl", "annotations/test.jsonl", "captions/test.jsonl"))
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
                    raise FileNotFoundError(f"missing TACoS video {video}")
                duration = float(record.get("duration") or 0)
                if duration <= 0:
                    from ..media import probe_video

                    duration = probe_video(path).duration
                targets = tuple(
                    (max(0.0, float(value[0])), min(duration, float(value[1])))
                    for value in record.get("relevant_windows", [])
                    if len(value) >= 2 and float(value[1]) > float(value[0])
                )
                sample = Sample(
                    id=str(record.get("qid", f"test:{line_number}")),
                    video=video,
                    video_path=path,
                    duration=duration,
                    query=str(record["query"]).strip(),
                    targets=targets,
                    group="tacos",
                )
                sample.validate()
                rows.append(sample)
        return rows

    def evaluate(self, records):
        return evaluate_records(records, multi_span=False)
