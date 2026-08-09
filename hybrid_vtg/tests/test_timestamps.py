import pytest

from hybrid_vtg.timestamps import (
    consolidate_intervals,
    normalize_timestamp,
    parse_intervals,
    parse_intervals_detailed,
    parse_timestamp,
)
from hybrid_vtg.types import Component


def test_parse_json_timestamp():
    assert parse_timestamp('```json\n{"start": 12.5, "end": 18}\n```') == (12.5, 18.0)


def test_parse_sentence_uses_final_span():
    assert parse_timestamp("clip 0 to 20; final answer 4.2s-7.9s") == (4.2, 7.9)


def test_parse_multispan_json_array():
    assert parse_intervals("[[4.0, 5.0], [1.0, 2.0], [1.0, 2.0]]") == (
        (1.0, 2.0), (4.0, 5.0),
    )


def test_parse_multispan_object_arrays_and_start_time_aliases():
    text = '''```json
    [{"start": 4, "end": 5}, {"start_time": 1, "end_time": 2}]
    ```'''
    parsed = parse_intervals_detailed(text)
    assert parsed.intervals == ((1.0, 2.0), (4.0, 5.0))
    assert parsed.status == "valid_json"


def test_parse_recovers_complete_objects_from_truncated_generation():
    parsed = parse_intervals_detailed(
        '[{"start": 1, "end": 2}, {"start": 4, "end": 5}, {"start": 8'
    )
    assert parsed.intervals == ((1.0, 2.0), (4.0, 5.0))
    assert parsed.status == "recovered"


def test_parse_distinguishes_explicit_empty_from_invalid_text():
    assert parse_intervals_detailed("```json\n[]\n```").status == "explicit_empty"
    with pytest.raises(ValueError, match="contains no interval set"):
        parse_intervals_detailed("I cannot determine the interval")


def test_shared_interval_consolidation_suppresses_duplicates_and_merges_gaps():
    assert consolidate_intervals(
        ((1.0, 5.0), (1.2, 4.8), (5.5, 7.0), (20.0, 25.0)),
        duration=22.0,
    ) == ((1.2, 7.0), (20.0, 22.0))


def test_relative_timestamp_is_converted_and_clamped():
    component = Component(100.0, 120.0, 0.8)
    assert normalize_timestamp((3.0, 8.0), component, 200.0, "relative") == (103.0, 108.0)


def test_invalid_normalized_span_fails_instead_of_using_annotation():
    component = Component(10.0, 20.0, 0.8)
    with pytest.raises(ValueError):
        normalize_timestamp((30.0, 40.0), component, 100.0, "absolute")
