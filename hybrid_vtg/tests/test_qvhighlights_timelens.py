import json
from pathlib import Path

import pytest

from hybrid_vtg.benchmarks.qvhighlights_timelens import QVHighlightsTimeLensBenchmark
from hybrid_vtg.registry import BENCHMARKS, load_builtin_plugins


def test_qvhighlights_timelens_registered():
    load_builtin_plugins()
    assert "qvhighlights-timelens" in BENCHMARKS.names()
    benchmark = BENCHMARKS.create("qvhighlights-timelens")
    assert isinstance(benchmark, QVHighlightsTimeLensBenchmark)


def test_qvhighlights_timelens_loads_samples(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "vid1.mp4").touch()
    (tmp_path / "videos" / "vid2.mp4").touch()

    data = {
        "vid1": {
            "duration": 150.0,
            "spans": [[0.0, 80.0]],
            "queries": ["A girl in a red top is speaking to the camera."],
        },
        "vid2": {
            "duration": 150.0,
            "spans": [[10.0, 20.0], [50.0, 60.0]],
            "queries": ["First event query", "Second event query"],
        },
    }
    (tmp_path / "qvhighlights-timelens.json").write_text(json.dumps(data), encoding="utf-8")

    benchmark = QVHighlightsTimeLensBenchmark()
    samples = benchmark.load_test(tmp_path)

    assert len(samples) == 3
    assert samples[0].id == "vid1::0"
    assert samples[0].video == "vid1"
    assert samples[0].duration == 150.0
    assert samples[0].query == "A girl in a red top is speaking to the camera."
    assert samples[0].targets == ((0.0, 80.0),)
    assert samples[0].cardinality == "single"

    assert samples[1].id == "vid2::0"
    assert samples[1].targets == ((10.0, 20.0),)
    assert samples[2].id == "vid2::1"
    assert samples[2].targets == ((50.0, 60.0),)


def test_qvhighlights_timelens_evaluates_metrics():
    benchmark = QVHighlightsTimeLensBenchmark()
    records = [
        {
            "targets": [[10.0, 20.0]],
            "prediction": {"spans": [{"start": 10.0, "end": 20.0, "score": 1.0}]},
        },
        {
            "targets": [[30.0, 40.0]],
            "prediction": {"spans": [{"start": 30.0, "end": 50.0, "score": 1.0}]},  # IoU = 10/20 = 0.5
        },
    ]
    metrics = benchmark.evaluate(records)
    assert metrics is not None
    assert metrics["count"] == 2
    assert metrics["mIoU"] == pytest.approx(0.75)
    assert metrics["R@1,IoU=0.3"] == pytest.approx(1.0)
    assert metrics["R@1,IoU=0.5"] == pytest.approx(1.0)
    assert metrics["R@1,IoU=0.7"] == pytest.approx(0.5)
