from hybrid_vtg.optimization_validation import compare_optimization_runs


def _record(sample_id: str, interval=(1.0, 2.0), total=2.0):
    return {
        "id": sample_id,
        "targets": [[1.0, 2.0]],
        "prediction": {
            "interval": list(interval),
            "telemetry": {"first_token_topk": {"token_ids": [1, 2], "logits": [4.0, 3.0]}},
        },
        "efficiency": {"timing_seconds": {"total": total}, "qwen_oom_fallbacks": 0},
    }


def test_equivalence_gate_checks_intervals_logits_and_speed():
    baseline = [_record(str(index), total=2.0) for index in range(4)]
    candidate = [_record(str(index), total=1.5) for index in range(4)]
    result = compare_optimization_runs(
        baseline, candidate, mode="equivalence", minimum_samples=4, minimum_speedup=0.2,
    )
    assert result["passed"]


def test_refinement_gate_enforces_accuracy_budget():
    baseline = [_record(str(index)) for index in range(4)]
    candidate = [_record(str(index), interval=(0.0, 3.0)) for index in range(4)]
    result = compare_optimization_runs(
        baseline, candidate, mode="refinement", minimum_samples=4,
    )
    assert not result["passed"]
    assert not result["accuracy_pass"]
