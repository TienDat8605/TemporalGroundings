import numpy as np
import pytest

from hybrid_vtg.scout_features import annotation_queries, model_slug, sample_timestamps


def test_sample_timestamps_uses_temporal_cell_centers():
    assert np.array_equal(sample_timestamps(3.0, 1.0), np.asarray([0.5, 1.5, 2.5], dtype=np.float32))
    assert np.array_equal(sample_timestamps(0.25, 1.0), np.asarray([np.nextafter(0.25, 0.0)], dtype=np.float32))


@pytest.mark.parametrize("duration,fps", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_sample_timestamps_rejects_nonpositive_values(duration: float, fps: float):
    with pytest.raises(ValueError):
        sample_timestamps(duration, fps)


def test_annotation_queries_matches_existing_scout_id_schema():
    annotation = {
        "video-a": {"queries": ["first query", "second query"]},
        "video-b": {"queries": ["third query"]},
    }
    ids, queries = annotation_queries(annotation)
    assert ids == ["video-a::0", "video-a::1", "video-b::0"]
    assert queries == ["first query", "second query", "third query"]


def test_model_slug_matches_existing_archive_layout():
    assert model_slug("nvidia/llama-nemotron-embed-vl-1b-v2") == "nvidia--llama-nemotron-embed-vl-1b-v2"
