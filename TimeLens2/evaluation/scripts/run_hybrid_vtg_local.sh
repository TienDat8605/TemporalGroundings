#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_DIR="$(cd "${EVALUATION_DIR}/.." && pwd)"

DATASET="${DATASET:-charades-sta}"
SPLIT="${SPLIT:-auto}"
RUN_NAME="${RUN_NAME:-hybrid-training-free}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TEMPORAL_POLICY="${TEMPORAL_POLICY:-hybrid}"
SPATIAL_POLICY="${SPATIAL_POLICY:-hybrid}"
SPATIAL_KEEP_RATIO="${SPATIAL_KEEP_RATIO:-0.25}"
GROUNDER_BACKEND="${GROUNDER_BACKEND:-proposal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_DIR}/../results/hybrid_vtg}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/timelens2-hybrid}"

cd "${EVALUATION_DIR}"
python run_hybrid_vtg.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --run-name "${RUN_NAME}" \
  --max-samples "${MAX_SAMPLES}" \
  --temporal-policy "${TEMPORAL_POLICY}" \
  --spatial-policy "${SPATIAL_POLICY}" \
  --spatial-keep-ratio "${SPATIAL_KEEP_RATIO}" \
  --grounder-backend "${GROUNDER_BACKEND}" \
  --output-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  "$@"
