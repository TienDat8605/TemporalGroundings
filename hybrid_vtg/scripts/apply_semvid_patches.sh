#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
semvid_root="${SEMVID_ROOT:-$repo_root/SemVID}"
patch_files=(
  "$repo_root/hybrid_vtg/patches/semvid-qwen-microbatch.patch"
  "$repo_root/hybrid_vtg/patches/semvid-tpsa-selection-hook.patch"
)

for patch_file in "${patch_files[@]}"; do
  if git -C "$semvid_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    continue
  fi
  git -C "$semvid_root" apply --check "$patch_file"
  git -C "$semvid_root" apply "$patch_file"
done
