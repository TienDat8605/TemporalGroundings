#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${TACOS_ROOT:?Set TACOS_ROOT to the TACoS dataset root}"
tacos_split="${TACOS_SPLIT:-test}"

bash "$repo_root/hybrid_vtg/scripts/apply_semvid_patches.sh"

hybrid-vtg run \
  --benchmark tacos \
  --data "$TACOS_ROOT" \
  --split "$tacos_split" \
  --output "$repo_root/outputs/hybrid-vtg/tacos-${tacos_split}.jsonl" \
  --cache-dir "$repo_root/.cache/hybrid-vtg" \
  "$@"
