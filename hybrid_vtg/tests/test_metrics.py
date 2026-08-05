from hybrid_vtg.metrics import evaluate, temporal_iou


def test_temporal_iou_and_summary():
    assert temporal_iou((0, 5), (2, 7)) == 3 / 7
    result = evaluate([{
        "targets": [[2, 7]],
        "prediction": {"interval": [0, 5]},
    }])
    assert result["count"] == 1
    assert result["R@1,IoU=0.3"] == 1.0
    assert result["R@1,IoU=0.5"] == 0.0
