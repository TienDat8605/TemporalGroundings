"""One predictable result tree and its generated human-readable index."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def run_directory(root: Path, benchmark: str, model: str, method: str, seed: int) -> Path:
    return root / "runs" / benchmark / model / method / f"seed-{seed}"


def refresh_results_index(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "runs").glob("*/*/*/seed-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = manifest_path.parent
        for metrics_path in sorted((run_dir / "metrics").glob("p*.json")):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "benchmark": manifest["benchmark"],
                    "model": manifest.get("result_model", manifest["model"]),
                    "method": manifest.get("result_method", manifest["method"]),
                    "seed": manifest["seed"],
                    "subset": metrics_path.stem.removeprefix("p"),
                    "requested": metrics.get("requested", 0),
                    "successful": metrics.get("successful", 0),
                    "failed": metrics.get("failed", 0),
                    "metrics": json.dumps(metrics.get("metrics"), sort_keys=True),
                    "run": str(run_dir.relative_to(root)),
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    columns = (
        "benchmark",
        "model",
        "method",
        "seed",
        "subset",
        "requested",
        "successful",
        "failed",
        "metrics",
        "run",
    )
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Hybrid VTG results",
        "",
        "This file is generated from run manifests and metrics. QVHighlights test runs are",
        "submission-only because the official labels are hidden.",
        "",
        "| Benchmark | Model | Method | Seed | Subset | Success | Metrics |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['model']} | {row['method']} | {row['seed']} | "
            f"{row['subset']}% | {row['successful']}/{row['requested']} | `{row['metrics']}` |"
        )
    if not rows:
        lines.append("| — | — | — | — | — | — | No new-schema runs yet |")
    lines.extend(
        [
            "",
            "Historical artifacts retained during the refactor are under `legacy/`; their",
            "completeness and provenance are documented in `legacy/README.md`.",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
