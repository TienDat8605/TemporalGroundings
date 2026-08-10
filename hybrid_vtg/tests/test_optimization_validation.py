from hybrid_vtg.optimization_validation import (
    compare_optimization_runs, validate_tpsa_promotion, validate_tpsa_v3_gate,
)


def _record(sample_id: str, interval=(1.0, 2.0), total=2.0):
    return {
        "id": sample_id,
        "targets": [[1.0, 2.0]],
        "prediction": {
            "interval": list(interval),
            "telemetry": {"first_token_topk": {"token_ids": [1, 2], "logits": [4.0, 3.0]}},
        },
        "efficiency": {"total_seconds": total, "qwen_oom_fallback": False},
    }


def test_equivalence_gate_checks_intervals_logits_and_speed():
    baseline = [_record(str(index), total=2.0) for index in range(4)]
    candidate = [_record(str(index), total=1.5) for index in range(4)]
    result = compare_optimization_runs(
        baseline, candidate, minimum_samples=4, minimum_speedup=0.2,
    )
    assert result["passed"]


def test_tpsa_promotion_requires_two_equal_compute_dataset_wins():
    def rows(candidate: bool):
        output = []
        for index in range(20):
            interval = (1.0, 2.0) if candidate or index >= 2 else (0.0, 3.0)
            row = _record(str(index), interval=interval, total=2.1 if candidate else 2.0)
            row["efficiency"].update({"decoded_frames": 32, "actual_retained_tokens": 128})
            output.append(row)
        return output

    result = validate_tpsa_promotion(
        {"tacos": rows(False), "charades": rows(False)},
        {"tacos": rows(True), "charades": rows(True)},
    )
    assert result["passed"]
    assert result["improved_datasets"] == 2


def test_tpsa_v3_gate_enforces_metrics_compute_and_latency():
    control = [_record(str(index), total=6.4) for index in range(64)]
    candidate = [_record(str(index), total=6.7) for index in range(64)]
    for row in control:
        row["spatial_policy"] = "tpsa_query"
    for row in candidate:
        row["spatial_policy"] = "tpsa_boundary"
    for row in control + candidate:
        row["efficiency"].update({"decoded_frames": 32, "actual_retained_tokens": 128})
    control_metrics = {
        "EtF1": 38.585069, "tF1@0.7": 42.948909, "tIoU": 57.886244,
        "C-Acc": 50.0, "parse_rate": 1.0, "mean_end_to_end_seconds": 6.432,
    }
    candidate_metrics = {
        "EtF1": 39.7, "tF1@0.7": 43.0, "tIoU": 57.9,
        "C-Acc": 48.4375, "parse_rate": 1.0, "mean_end_to_end_seconds": 6.7,
    }
    result = validate_tpsa_v3_gate(
        control, candidate, control_metrics, candidate_metrics,
    )
    assert result["passed"]
    candidate_metrics["mean_end_to_end_seconds"] = 6.755
    assert not validate_tpsa_v3_gate(
        control, candidate, control_metrics, candidate_metrics,
    )["passed"]
