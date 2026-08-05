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
