import pytest

from hybrid_vtg.timestamps import normalize_timestamp, parse_grounding_response, parse_timestamp
from hybrid_vtg.types import Component


def test_parse_json_timestamp():
    assert parse_timestamp('```json\n{"start": 12.5, "end": 18}\n```') == (12.5, 18.0)


def test_parse_sentence_uses_final_span():
    assert parse_timestamp("clip 0 to 20; final answer 4.2s-7.9s") == (4.2, 7.9)


def test_parse_explicit_positive_grounding_response():
    response = parse_grounding_response(
        '{"present": true, "confidence": 0.8, "start": 12.5, "end": 18}'
    )
    assert response.present
    assert response.interval == (12.5, 18.0)
    assert response.confidence == 0.8


def test_parse_explicit_negative_grounding_response_without_timestamp():
    response = parse_grounding_response('{"present": false, "confidence": 0.9}')
    assert not response.present
    assert response.interval is None
    assert response.confidence == 0.9


def test_legacy_timestamp_response_remains_positive():
    response = parse_grounding_response('{"start": 12.5, "end": 18}')
    assert response.present
    assert response.interval == (12.5, 18.0)
    assert response.confidence == 1.0


def test_relative_timestamp_is_converted_and_clamped():
    component = Component(100.0, 120.0, 0.8)
    assert normalize_timestamp((3.0, 8.0), component, 200.0, "relative") == (103.0, 108.0)


def test_invalid_normalized_span_fails_instead_of_using_annotation():
    component = Component(10.0, 20.0, 0.8)
    with pytest.raises(ValueError):
        normalize_timestamp((30.0, 40.0), component, 100.0, "absolute")
