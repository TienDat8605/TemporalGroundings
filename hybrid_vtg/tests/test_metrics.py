from hybrid_vtg.metrics import endpoint_availability, evaluate, target_coverage, temporal_iou


def test_temporal_iou_and_summary():
    assert temporal_iou((0, 5), (2, 7)) == 3 / 7
    result = evaluate([{
        "targets": [[2, 7]],
        "prediction": {"interval": [0, 5]},
    }])
    assert result["count"] == 1
    assert result["R@1,IoU=0.3"] == 1.0
    assert result["R@1,IoU=0.5"] == 0.0


def test_router_metrics_measure_evidence_availability_not_proposal_iou():
    components = [{"start": 0.0, "end": 16.0}]
    target = [7.0, 9.0]
    assert target_coverage(components, target) == 1.0
    assert endpoint_availability(components, target) == (True, True, True)
    result = evaluate([{
        "targets": [target],
        "prediction": {"interval": target},
        "route": {"components": components, "retained_fraction": 0.5},
    }])
    assert result["RouterTargetCoverage@0.5"] == 1.0
    assert result["RouterFullContainment"] == 1.0
    assert result["RouterBothEndpointsAvailable"] == 1.0


def test_summary_reports_fallback_and_component_rejections():
    result = evaluate([{
        "targets": [[2.0, 4.0]],
        "prediction": {"interval": [2.0, 4.0], "presence_score": 0.8},
        "route": {
            "components": [{"start": 0.0, "end": 10.0}],
            "retained_fraction": 1.0,
            "low_confidence_fallback": True,
        },
        "component_predictions": [{"interval": [2.0, 4.0]}],
        "component_errors": [{"event_present": False}],
    }])
    assert result["TemporalFallbackRate"] == 1.0
    assert result["mean_routed_component_count"] == 1.0
    assert result["mean_component_rejection_fraction"] == 0.5
    assert result["mean_selected_presence_score"] == 0.8
