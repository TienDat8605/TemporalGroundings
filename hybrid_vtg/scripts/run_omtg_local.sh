#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OMTG_ROOT:?Set OMTG_ROOT to the prepared OMTG Bench root}"
bash "$repo_root/hybrid_vtg/scripts/apply_semvid_patches.sh"
hybrid-vtg run \
  --benchmark omtg \
  --data "$OMTG_ROOT" \
  --split test \
  --output "$repo_root/outputs/hybrid-vtg/omtg-test.jsonl" \
  "$@"
