from pathlib import Path

import pytest
import torch

from hybrid_vtg.contracts import Sample, TemporalEvidence
from hybrid_vtg.registry import BENCHMARKS, METHODS, MODELS, Registry, load_builtin_plugins
from hybrid_vtg.sampling import percentage_key, subset_samples


def sample(index: int) -> Sample:
    return Sample(str(index), f"v{index}", Path(__file__), 10.0, f"query {index}")


def test_builtin_surface_is_exactly_the_requested_matrix():
    load_builtin_plugins()
    assert METHODS.names() == (
        "coarse-to-fine-64",
        "native",
    )
    assert MODELS.names() == (
        "qwen2-vl-7b",
        "qwen3-vl-4b",
        "timelens-7b",
        "timelens-8b",
        "timelens2-4b",
        "unitime",
        "univtg",
    )
    assert BENCHMARKS.names() == ("omtg", "qvhighlights", "tacos")


def test_all_qwen3_backends_get_4096_evidence_cap(tmp_path):
    load_builtin_plugins()
    timelens2 = MODELS.create("timelens2-4b", cache_dir=tmp_path)
    timelens8 = MODELS.create("timelens-8b", cache_dir=tmp_path)
    qwen3 = MODELS.create("qwen3-vl-4b", cache_dir=tmp_path)
    assert timelens2.maximum_evidence_units == 4_096
    assert timelens8.maximum_evidence_units == 4_096
    assert qwen3.maximum_evidence_units == 4_096


def test_registry_rejects_duplicates_and_unknown_values():
    registry = Registry("thing")
    registry.register("one", lambda: 1)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("one", lambda: 2)
    with pytest.raises(ValueError, match="unknown"):
        registry.create("two")


def test_seeded_query_subsets_are_reproducible_and_nested():
    values = [sample(index) for index in range(23)]
    ten = subset_samples(values, 10, 42)
    twenty = subset_samples(values, 20, 42)
    assert len(ten) == 3
    assert len(twenty) == 5
    assert [value.id for value in ten] == [value.id for value in twenty[:3]]
    assert [value.id for value in twenty] != [value.id for value in subset_samples(values, 20, 43)]


def test_any_percentage_from_zero_through_one_hundred_is_supported():
    values = [sample(index) for index in range(23)]
    assert subset_samples(values, 0, 42) == []
    assert len(subset_samples(values, 12.5, 42)) == 3
    assert len(subset_samples(values, 37, 42)) == 9
    assert len(subset_samples(values, 100, 42)) == 23
    assert percentage_key(12.5) == "p012.5"
    assert percentage_key(100) == "p100"
    with pytest.raises(ValueError, match="between 0 and 100"):
        subset_samples(values, 100.1, 42)


def test_temporal_evidence_selection_and_merge_keep_timestamps():
    first = TemporalEvidence(
        torch.arange(6).reshape(3, 2),
        (0.0, 1.0, 2.0),
        3,
        {"cell_coordinates": [[0, 0, index] for index in range(3)]},
    )
    second = TemporalEvidence(
        torch.arange(4).reshape(2, 2),
        (0.5, 1.5),
        2,
        {"cell_coordinates": [[0, 1, index] for index in range(2)]},
    )
    selected = first.select([2, 0])
    assert selected.timestamps == (2.0, 0.0)
    assert selected.metadata["cell_coordinates"] == [[0, 0, 2], [0, 0, 0]]
    merged = TemporalEvidence.concatenate([first, second])
    assert merged.timestamps == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert len(merged.metadata["cell_coordinates"]) == merged.size
