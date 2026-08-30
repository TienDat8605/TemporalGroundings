#!/usr/bin/env python3
import json
from pathlib import Path

def evaluate_cardinality_stratification():
    results_dir = Path("results/omtg_100pct_adaptive_benchmarks")
    if not results_dir.is_dir():
        print("Results directory not found, skipping stratification.")
        return

    runs = {
        "Native (Paper Prompt)": results_dir / "timelens_8b_native_whole_video.json",
        "Adaptive SGDE-64": results_dir / "timelens_8b_adaptive_sgde_64f.json",
        "Adaptive SGDE-128": results_dir / "timelens_8b_adaptive_sgde_128f.json",
        "Adaptive SGDE-256": results_dir / "timelens_8b_adaptive_sgde_256f.json",
    }

    print("\n=== OMTG CARDINALITY STRATIFICATION REPORT ($K$-Split) ===")
    for name, path in runs.items():
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\nModel: {name}")
        print(f"Overall C-Acc: {data.get('count_accuracy', 0)*100:.2f}% | EtF1: {data.get('effective_tf1_0.5', 0)*100:.2f}%")

if __name__ == "__main__":
    evaluate_cardinality_stratification()
