import json
from pathlib import Path

import pytest

from hybrid_vtg.benchmarks.qvhighlights import QVHighlightsBenchmark
from hybrid_vtg.benchmarks.tacos import TACoSBenchmark
from hybrid_vtg.io import read_jsonl
from hybrid_vtg.metrics import evaluate_records, one_to_many_metrics


def test_tacos_loads_only_test(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "v1.mp4").touch()
    (tmp_path / "test.jsonl").write_text(
        json.dumps(
            {
                "qid": 7,
                "vid": "v1",
                "duration": 20,
                "query": "cut onion",
                "relevant_windows": [[2, 5], [7, 9]],
            }
        )
        + "\n"
    )
    values = TACoSBenchmark().load_test(tmp_path)
    assert len(values) == 1
    assert values[0].targets == ((2.0, 5.0), (7.0, 9.0))


def test_qvhighlights_submission_contains_only_moment_retrieval(tmp_path: Path):
    records = [
        {
            "id": "12",
            "query": "open door",
            "video": "v1",
            "prediction": {"spans": [{"start": 1, "end": 3, "score": 0.8}]},
        }
    ]
    path = QVHighlightsBenchmark().export_submission(records, tmp_path / "submission.jsonl")
    value = read_jsonl(path)[0]
    assert value["pred_relevant_windows"] == [[1.0, 3.0, 0.8]]
    assert "pred_saliency_scores" not in value


def test_single_span_metrics_use_best_reference_window():
    records = [
        {
            "targets": [[1, 2], [5, 8]],
            "prediction": {"spans": [{"start": 5, "end": 8, "score": 1.0}]},
        }
    ]
    metrics = evaluate_records(records, multi_span=False)
    assert metrics["mIoU"] == 1.0
    assert metrics["R@1,IoU=0.7"] == 1.0


def test_omtg_cardinality_and_effective_f1():
    values = one_to_many_metrics([[0, 2], [4, 6]], [[0, 2], [4, 6]])
    assert values["C-Acc"] == 1.0
    assert values["EtF1"] == pytest.approx(1.0)
