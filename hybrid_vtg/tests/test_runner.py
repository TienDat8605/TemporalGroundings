from pathlib import Path

import pytest

from hybrid_vtg.contracts import Sample
from hybrid_vtg.io import ensure_manifest
from hybrid_vtg.runner import _evaluation_summary, _pruning_variant, _validate_pruning_configuration, run_benchmark


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


def test_pruning_configurations_get_independent_result_directories():
    assert _pruning_variant("qwen", "none", 1.0, 0, "none", 1.0) == "qwen"
    assert (
        _pruning_variant("qwen", "mage", 0.5, 0, "semvid", 0.125)
        == "qwen--enc-mage-r0.5-l0--post-semvid-r0.125"
    )


def test_pruning_configuration_is_validated_before_a_run():
    with pytest.raises(ValueError, match="encoder retention requires"):
        _validate_pruning_configuration("qwen", "none", 0.5, 0, "none", 1.0)
    with pytest.raises(ValueError, match="Qwen-based"):
        _validate_pruning_configuration("univtg", "mage", 0.5, 0, "none", 1.0)


def test_adaptive_corridor_count_is_validated_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 8"):
        run_benchmark(
            benchmark_name="omtg",
            data=tmp_path,
            model_name="unitime",
            method_name="unitime-adaptive",
            percentage=1,
            seed=1,
            corridor_top_k=9,
        )


def test_manifest_mismatch_requires_rerun(tmp_path):
    path = tmp_path / "manifest.json"
    ensure_manifest(path, {"revision": "old"})

    with pytest.raises(RuntimeError, match="manifest differs"):
        ensure_manifest(path, {"revision": "new"})

    ensure_manifest(path, {"revision": "new"}, replace=True)
    assert path.read_text(encoding="utf-8") == '{\n  "revision": "new"\n}\n'
