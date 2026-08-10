"""OMTG Bench fixed 320-query test split."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from ..contracts import Benchmark, Sample
from ..metrics import evaluate_records
from ..postprocess import parse_spans
from .common import first_file, video_index

_QUERY = re.compile(r"textual query\s+['\"](?P<query>.+)['\"]\s+and determine", re.I)


class OMTGBenchmark(Benchmark):
    name = "omtg"

    def load_test(self, root: Path) -> list[Sample]:
        annotation = first_file(root, ("OMTGBench.tsv",))
        videos = video_index(root)
        rows = []
        with annotation.open(encoding="utf-8", newline="") as handle:
            for index, record in enumerate(csv.DictReader(handle, delimiter="\t")):
                video_name = str(record["video"])
                path = videos.get(Path(video_name).stem)
                if path is None:
                    raise FileNotFoundError(f"missing OMTG video {video_name}")
                targets = tuple((span.start, span.end) for span in parse_spans(str(record["answer"])))
                declared_duration = float(record.get("duration") or 0)
                if declared_duration > 0:
                    duration = max(declared_duration, max((end for _, end in targets), default=0))
                else:
                    from ..media import probe_video

                    duration = probe_video(path).duration
                question = str(record["question"]).strip()
                match = _QUERY.search(question)
                sample = Sample(
                    id=str(record.get("id", index)),
                    video=video_name,
                    video_path=path,
                    duration=duration,
                    query=match.group("query").strip() if match else question,
                    targets=targets,
                    group="omtg",
                    cardinality="multi",
                )
                sample.validate()
                rows.append(sample)
        return rows

    def evaluate(self, records):
        return evaluate_records(records, multi_span=True)
