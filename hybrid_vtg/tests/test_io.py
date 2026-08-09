from hybrid_vtg.io import completed_ids


def test_failed_attempt_is_complete_for_append_only_resume():
    assert completed_ids([
        {"id": "ok", "prediction": {"interval": [1.0, 2.0]}},
        {"id": "failed", "prediction": None, "error": "parse failure"},
    ]) == {"ok", "failed"}
