#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CHARADES_STA_ROOT:?Set CHARADES_STA_ROOT to the Charades-STA dataset root}"

bash "$repo_root/hybrid_vtg/scripts/apply_semvid_patches.sh"

hybrid-vtg run \
  --benchmark charades-sta \
  --data "$CHARADES_STA_ROOT" \
  --split test \
  --output "$repo_root/outputs/hybrid-vtg/charades-sta.jsonl" \
  --cache-dir "$repo_root/.cache/hybrid-vtg" \
  "$@"
