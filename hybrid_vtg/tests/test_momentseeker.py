import json
from pathlib import Path
from unittest.mock import patch

from hybrid_vtg.benchmarks.momentseeker import MomentSeekerBenchmark
from hybrid_vtg.registry import BENCHMARKS, load_builtin_plugins


def test_momentseeker_registered():
    load_builtin_plugins()
    assert "momentseeker" in BENCHMARKS.names()
    benchmark = BENCHMARKS.create("momentseeker")
    assert isinstance(benchmark, MomentSeekerBenchmark)


def test_momentseeker_loads_samples(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "ego_79.mp4").touch()
    (tmp_path / "videos" / "sport_12.mp4").touch()

    data = [
        {
            "qry_text": "The video shows a man playing a game of Connect Four with a woman.",
            "qry_img_path": "",
            "qry_video_path": "",
            "src_video_path": "./videos/ego_79.mp4",
            "task": "Description Location",
            "answering_time_interval": [[415.0, 480.0]],
        },
        {
            "qry_text": "A player scores a goal.",
            "qry_img_path": "",
            "qry_video_path": "",
            "src_video_path": "./videos/sport_12.mp4",
            "task": "Action Recognition",
            "answering_time_interval": [[10.0, 25.0], [80.0, 95.0]],
        },
    ]
    (tmp_path / "t2v.json").write_text(json.dumps(data), encoding="utf-8")

    with patch("hybrid_vtg.media.probe_video") as mock_probe:
        mock_probe.return_value.duration = 600.0

        benchmark = MomentSeekerBenchmark()
        samples = benchmark.load_test(tmp_path)

    assert len(samples) == 2
    assert samples[0].id == "ms_0000::ego_79"
    assert samples[0].video == "ego_79"
    assert samples[0].duration == 600.0
    assert samples[0].query == "The video shows a man playing a game of Connect Four with a woman."
    assert samples[0].targets == ((415.0, 480.0),)
    assert samples[0].group == "Description Location"
    assert samples[0].cardinality == "single"

    assert samples[1].id == "ms_0001::sport_12"
    assert samples[1].targets == ((10.0, 25.0), (80.0, 95.0))
    assert samples[1].group == "Action Recognition"
    assert samples[1].cardinality == "multi"


def test_momentseeker_evaluates_metrics():
    benchmark = MomentSeekerBenchmark()
    records = [
        {
            "targets": [[10.0, 20.0]],
            "prediction": {"spans": [{"start": 10.0, "end": 20.0, "score": 1.0}]},
        },
        {
            "targets": [[50.0, 60.0]],
            "prediction": {"spans": [{"start": 0.0, "end": 10.0, "score": 1.0}]},
        },
    ]
    metrics = benchmark.evaluate(records)
    assert "mIoU" in metrics
    assert "R@1,IoU=0.3" in metrics
    assert "R@1,IoU=0.5" in metrics
    assert "R@1,IoU=0.7" in metrics
    assert metrics["mIoU"] == 0.5
    assert metrics["R@1,IoU=0.3"] == 0.5
    assert metrics["R@1,IoU=0.5"] == 0.5
    assert metrics["R@1,IoU=0.7"] == 0.5
