#!/usr/bin/env python3
"""Extract resumable image/query embeddings for SGDE scouting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_vtg.scout_features import (
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    extract_scout_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("assets/datasets/qvhighlights-timelens"))
    parser.add_argument("--output-root", type=Path, default=Path("assets/features/scouts"))
    parser.add_argument("--model", dest="model_id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--max-input-tiles", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--archive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(extract_scout_features(**vars(parse_args())), indent=2))
