"""MomentSeeker Text-to-Moment benchmark adapter (avery00/MomentSeeker)."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import Benchmark, Sample
from ..metrics import evaluate_records
from .common import first_file, video_index


class MomentSeekerBenchmark(Benchmark):
    name = "momentseeker"

    def load_test(self, root: Path) -> list[Sample]:
        annotation = first_file(
            root,
            (
                "t2v.json",
                "annotations/t2v.json",
                "metadata/t2v.json",
            ),
        )
        videos = video_index(root)
        with annotation.open(encoding="utf-8") as handle:
            data = json.load(handle)

        from ..media import probe_video

        # Cache probed durations to avoid probing the same video files repeatedly
        duration_cache: dict[str, float] = {}

        rows = []
        for index, record in enumerate(data):
            src_video_path = record.get("src_video_path", "")
            vid = Path(src_video_path).stem
            path = videos.get(vid)
            if path is None:
                # Also try matching filename directly
                filename = Path(src_video_path).name
                candidate = root / "videos" / filename
                if candidate.is_file():
                    path = candidate
                else:
                    raise FileNotFoundError(f"missing MomentSeeker video: {src_video_path}")

            if vid not in duration_cache:
                duration_cache[vid] = probe_video(path).duration
            duration = duration_cache[vid]

            raw_intervals = record.get("answering_time_interval", [])
            targets = tuple(
                (max(0.0, float(span[0])), min(duration, float(span[1])))
                for span in raw_intervals
                if len(span) >= 2 and float(span[1]) > float(span[0])
            )
            task = record.get("task", "momentseeker")
            cardinality = "single" if len(targets) <= 1 else "multi"

            sample = Sample(
                id=f"ms_{index:04d}::{vid}",
                video=vid,
                video_path=path,
                duration=duration,
                query=str(record.get("qry_text", "")).strip(),
                targets=targets,
                group=str(task),
                cardinality=cardinality,
            )
            sample.validate()
            rows.append(sample)
        return rows

    def evaluate(self, records):
        return evaluate_records(records, multi_span=False)


__all__ = ["MomentSeekerBenchmark"]
