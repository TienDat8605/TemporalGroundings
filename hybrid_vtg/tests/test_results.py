import json
from pathlib import Path

from hybrid_vtg.results import refresh_results_index, run_directory


def test_result_layout_and_generated_note(tmp_path: Path):
    run = run_directory(tmp_path, "omtg", "qwen3-vl-4b", "coarse-to-fine-64", 42)
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "omtg",
                "model": "qwen3-vl-4b",
                "method": "coarse-to-fine-64",
                "seed": 42,
            }
        )
    )
    (run / "metrics-p010.json").write_text(
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
    assert "coarse-to-fine-64" in note


def test_hyperparameters_are_kept_in_flat_result_file(tmp_path: Path):
    hyperparameters = {"encoder_pruning": "mage", "encoder_retention": 0.5}
    run = run_directory(
        tmp_path,
        "omtg",
        "qwen3-vl-4b",
        "coarse-to-fine-64",
        7,
        hyperparameters=hyperparameters,
    )
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "omtg",
                "model": "qwen3-vl-4b",
                "method": "coarse-to-fine-64",
                "seed": 7,
            }
        )
    )
    (run / "metrics-p100.json").write_text(
        json.dumps(
            {
                "requested": 1,
                "successful": 1,
                "failed": 0,
                "metrics": {},
                "hyperparameters": hyperparameters,
            }
        )
    )

    refresh_results_index(tmp_path)

    assert run.parent == tmp_path / "runs"
    assert not (run / "metrics").exists()
    assert hyperparameters == json.loads((run / "metrics-p100.json").read_text())["hyperparameters"]
    assert "qwen3-vl-4b" in (tmp_path / "RESULTS.md").read_text()


def test_flat_run_directories_still_separate_hyperparameter_configurations(tmp_path: Path):
    common = (tmp_path, "omtg", "qwen3-vl-4b", "coarse-to-fine-64", 42)
    dense = run_directory(*common, hyperparameters={"encoder_pruning": "none"})
    mage = run_directory(*common, hyperparameters={"encoder_pruning": "mage"})

    assert dense != mage
    assert dense.parent == mage.parent == tmp_path / "runs"
