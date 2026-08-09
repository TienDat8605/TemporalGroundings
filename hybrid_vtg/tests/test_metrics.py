from hybrid_vtg.metrics import evaluate, one_to_many_metrics, temporal_iou


def test_temporal_iou_and_single_span_summary():
    assert temporal_iou((0, 5), (2, 7)) == 3 / 7
    result = evaluate([{
        "targets": [[2, 7]],
        "prediction": {"interval": [0, 5]},
    }])
    assert result["count"] == 1
    assert result["R@1,IoU=0.3"] == 1.0
    assert result["R@1,IoU=0.5"] == 0.0


def test_single_span_parse_failure_is_not_dropped():
    result = evaluate([{
        "duration": 8.0, "targets": [[2.0, 7.0]], "prediction": None,
    }])
    assert result["count"] == 1
    assert result["parsed_predictions"] == 0
    assert result["mIoU"] == 0.0
    assert result["boundary_MAE_seconds"] == 8.0


def test_omtg_exact_multispan_metrics():
    intervals = [[1.0, 2.0], [4.0, 5.0]]
    metrics = one_to_many_metrics(intervals, intervals)
    assert metrics["C-Acc"] == 1.0
    assert metrics["EtF1"] == 1.0
    assert metrics["tIoU"] == 1.0


def test_omtg_evaluation_uses_set_valued_predictions():
    result = evaluate([{
        "group": "omtg",
        "cardinality": "multi",
        "targets": [[1.0, 2.0], [4.0, 5.0]],
        "prediction": {"intervals": [[1.0, 2.0], [4.0, 5.0]]},
    }])
    assert result["C-Acc"] == 100.0
    assert result["EtF1"] == 100.0


def test_omtg_cardinality_errors_penalize_effective_f1():
    metrics = one_to_many_metrics([[1.0, 2.0]], [[1.0, 2.0], [4.0, 5.0]])
    assert metrics["C-Acc"] == 0.0
    assert metrics["CardinalityError"] == 1.0
    assert metrics["EtF1"] == 0.0


def test_omtg_parse_failures_count_as_empty_predictions():
    result = evaluate([{
        "group": "omtg", "cardinality": "multi", "targets": [[1.0, 2.0]],
        "prediction": None, "error": "unparseable",
    }])
    assert result["count"] == 1
    assert result["parsed_predictions"] == 0
    assert result["EtF1"] == 0.0


def test_omtg_evaluation_recovers_legacy_object_output_and_reports_status():
    result = evaluate([{
        "group": "omtg",
        "cardinality": "multi",
        "duration": 10.0,
        "targets": [[1.0, 2.0], [4.0, 5.0]],
        "prediction": {
            "intervals": [],
            "raw_text": '[{"start": 1, "end": 2}, {"start": 4, "end": 5}]',
        },
    }])
    assert result["C-Acc"] == 100.0
    assert result["EtF1"] == 100.0
    assert result["parse_rate"] == 1.0
    assert result["parse_status_counts"] == {"valid_json": 1}


def test_omtg_invalid_legacy_output_reduces_parse_rate():
    result = evaluate([{
        "group": "omtg", "cardinality": "multi", "duration": 10.0,
        "targets": [[1.0, 2.0]],
        "prediction": {"intervals": [], "raw_text": "not a timestamp"},
    }])
    assert result["parsed_predictions"] == 0
    assert result["parse_rate"] == 0.0
    assert result["parse_status_counts"] == {"invalid": 1}
