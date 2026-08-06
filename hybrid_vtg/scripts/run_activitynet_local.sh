#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ACTIVITYNET_ROOT:?Set ACTIVITYNET_ROOT to the ActivityNet-Grounding dataset root}"
activitynet_split="${ACTIVITYNET_SPLIT:-val_2}"
output_split="${activitynet_split//[^[:alnum:]._-]/_}"

bash "$repo_root/hybrid_vtg/scripts/apply_semvid_patches.sh"

hybrid-vtg run \
  --benchmark activitynet-grounding \
  --data "$ACTIVITYNET_ROOT" \
  --split "$activitynet_split" \
  --output "$repo_root/outputs/hybrid-vtg/activitynet-${output_split}.jsonl" \
  --cache-dir "$repo_root/.cache/hybrid-vtg" \
  "$@"
