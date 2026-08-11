import json
from pathlib import Path

from hybrid_vtg.results import refresh_results_index, run_directory


def test_result_layout_and_generated_note(tmp_path: Path):
    run = run_directory(tmp_path, "omtg", "qwen3-vl-4b", "tpsa-query", 42)
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "omtg",
                "model": "qwen3-vl-4b",
                "method": "tpsa-query",
                "seed": 42,
            }
        )
    )
    (run / "metrics").mkdir()
    (run / "metrics" / "p010.json").write_text(
        json.dumps(
            {
                "requested": 32,
                "successful": 31,
                "failed": 1,
                "metrics": {"EtF1": 12.3},
            }
        )
    )
    refresh_results_index(tmp_path)
    assert (tmp_path / "index.csv").is_file()
    note = (tmp_path / "RESULTS.md").read_text()
    assert "31/32" in note
    assert "tpsa-query" in note


def test_pruned_result_uses_its_configuration_name(tmp_path: Path):
    variant = "qwen3-vl-4b--enc-mage-r0.5-l0"
    run = run_directory(tmp_path, "omtg", variant, "hmve", 7)
    (run / "metrics").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "omtg",
                "model": "qwen3-vl-4b",
                "result_model": variant,
                "method": "hmve",
                "seed": 7,
            }
        )
    )
    (run / "metrics" / "p100.json").write_text(
        json.dumps({"requested": 1, "successful": 1, "failed": 0, "metrics": {}})
    )

    refresh_results_index(tmp_path)

    assert variant in (tmp_path / "index.csv").read_text()
    assert variant in (tmp_path / "RESULTS.md").read_text()
