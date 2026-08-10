from pathlib import Path

from hybrid_vtg.contracts import Sample
from hybrid_vtg.runner import _evaluation_summary


class RecordingBenchmark:
    def __init__(self):
        self.records = None

    def evaluate(self, records):
        self.records = records
        return {"C-Acc": 0.0}


def sample(sample_id="1"):
    return Sample(sample_id, "video", Path("video.mp4"), 10.0, "event", ((1.0, 2.0),), cardinality="multi")


def test_all_failed_run_does_not_publish_zero_metrics():
    benchmark = RecordingBenchmark()
    summary = _evaluation_summary(benchmark, [sample()], [])

    assert summary["status"] == "failed"
    assert summary["metrics"] is None
    assert benchmark.records is None


def test_partial_run_marks_failed_samples_as_empty_predictions():
    benchmark = RecordingBenchmark()
    records = [{"id": "1", "prediction": {"spans": []}}]
    summary = _evaluation_summary(benchmark, [sample("1"), sample("2")], records)

    assert summary["status"] == "partial"
    assert summary["successful"] == 1
    assert summary["failed"] == 1
    assert benchmark.records[1]["prediction"]["spans"] == []
