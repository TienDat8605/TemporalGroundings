#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ACTIVITYNET_ROOT:?Set ACTIVITYNET_ROOT to the ActivityNet-Grounding dataset root}"

hybrid-vtg run \
  --benchmark activitynet-grounding \
  --data "$ACTIVITYNET_ROOT" \
  --split val_2 \
  --output "$repo_root/outputs/hybrid-vtg/activitynet-val-2.jsonl" \
  --cache-dir "$repo_root/.cache/hybrid-vtg" \
  "$@"
