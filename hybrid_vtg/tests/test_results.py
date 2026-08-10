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
