#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${HYBRID_VTG_ASSET_ROOT:-$PROJECT_ROOT/assets}"
CONDA_ENV="${HYBRID_VTG_CONDA_ENV:-ML-study}"

cd "$PROJECT_ROOT"

if ! conda run -n "$CONDA_ENV" aria2c --version >/dev/null 2>&1; then
  echo "aria2c is required in conda environment '$CONDA_ENV'" >&2
  exit 1
fi

conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH=src \
  python scripts/download_assets.py qvhighlights-timelens \
  --root "$ASSET_ROOT" \
  --accept-licenses

conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH=src \
  python scripts/extract_scout_features.py \
  --dataset-root "$ASSET_ROOT/datasets/qvhighlights-timelens" \
  --output-root "$ASSET_ROOT/features/scouts" \
  --archive \
  "$@"
