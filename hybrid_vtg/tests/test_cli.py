from pathlib import Path

from hybrid_vtg import cli


def test_run_defaults_to_downloaded_benchmark_directory(monkeypatch, capsys):
    received = {}

    def fake_run_benchmark(**values):
        received.update(values)
        return {"failed": 0}

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
    assert (
        cli.main(
            [
                "run",
                "--benchmark",
                "omtg",
                "--model",
                "qwen3-vl-4b",
                "--method",
                "coarse-to-fine-64",
                "--subset",
                "10",
                "--seed",
                "42",
            ]
        )
        == 0
    )
    assert received["data"] == Path("assets/datasets/omtg")
    assert received["encoder_pruning"] == "none"
    assert received["encoder_retention"] == 1.0
    assert received["post_pruning"] == "none"
    assert received["post_retention"] == 1.0
    assert received["corridor_top_k"] == 4
    assert received["base_checkpoint"] is None
    assert '"failed": 0' in capsys.readouterr().out
