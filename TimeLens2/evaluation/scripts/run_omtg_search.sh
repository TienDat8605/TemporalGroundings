#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'error: NVIDIA GPU is required\n' >&2
  exit 2
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
if [[ "$GPU_NAME" != *T4* ]]; then
  printf 'error: this experiment is pinned to one Colab T4; detected %s\n' "$GPU_NAME" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  printf 'error: expected exactly one GPU\n' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export TIMELENS2_ATTN_IMPLEMENTATION="${TIMELENS2_ATTN_IMPLEMENTATION:-sdpa}"
export VLM_VIDEO_DECODE_BACKEND="${VLM_VIDEO_DECODE_BACKEND:-pyav}"
export TOKENIZERS_PARALLELISM=false

PHASE="${OMTG_PHASE:-all}"
RUN_NAME="${OMTG_RUN_NAME:-smoke}"
MAX_SAMPLES="${OMTG_MAX_SAMPLES:-25}"
BUDGETS="${OMTG_BUDGETS:-32,64}"
SCHEDULES="${OMTG_SCHEDULES:-uniform-one-shot,full-video-multipass,uniform-window-local,embedding-window-local}"
SESSION_OUTPUT_ROOT="${TIMELENS2_OMTG_OUTPUT_ROOT:-/content/timelens2-experiment-outputs/omtg_search}"
FETCH_OUTPUT_ROOT="$PWD/outputs/omtg_search"
CHECKPOINT_SECONDS="${OMTG_CHECKPOINT_SECONDS:-30}"
CHECKPOINT_PID=""

sync_outputs() {
  if [[ -d "$SESSION_OUTPUT_ROOT" ]]; then
    mkdir -p "$FETCH_OUTPUT_ROOT"
    cp -a "$SESSION_OUTPUT_ROOT/." "$FETCH_OUTPUT_ROOT/"
  fi
}

checkpoint_outputs() {
  [[ -n "${TIMELENS2_COLAB_RUN_DIR:-}" ]] || return 0
  [[ -d "$SESSION_OUTPUT_ROOT" ]] || return 0

  local checkpoint_path="$TIMELENS2_COLAB_RUN_DIR/omtg_checkpoint.tar.gz"
  local checkpoint_tmp="$TIMELENS2_COLAB_RUN_DIR/.omtg_checkpoint.tar.gz.tmp"
  local snapshot_dir
  local result=0
  snapshot_dir="$(mktemp -d "$TIMELENS2_COLAB_RUN_DIR/.omtg-checkpoint.XXXXXX")"
  cp -a "$SESSION_OUTPUT_ROOT/." "$snapshot_dir/" || result=$?
  if (( result == 0 )); then
    tar -czf "$checkpoint_tmp" -C "$snapshot_dir" . || result=$?
  fi
  if (( result == 0 )); then
    mv -f "$checkpoint_tmp" "$checkpoint_path"
  fi
  rm -rf -- "$snapshot_dir"
  if (( result != 0 )); then
    rm -f -- "$checkpoint_tmp"
  fi
  return "$result"
}

checkpoint_loop() {
  while sleep "$CHECKPOINT_SECONDS"; do
    checkpoint_outputs || true
  done
}

cleanup() {
  if [[ -n "$CHECKPOINT_PID" ]]; then
    kill "$CHECKPOINT_PID" >/dev/null 2>&1 || true
    wait "$CHECKPOINT_PID" 2>/dev/null || true
  fi
  checkpoint_outputs || true
  sync_outputs || true
}

[[ "$CHECKPOINT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || { printf 'error: OMTG_CHECKPOINT_SECONDS must be a positive integer\n' >&2; exit 2; }

mkdir -p "$SESSION_OUTPUT_ROOT"
trap cleanup EXIT
checkpoint_loop &
CHECKPOINT_PID=$!

python run_omtg_search.py \
  --phase "$PHASE" \
  --output-root "$SESSION_OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --max-samples "$MAX_SAMPLES" \
  --budgets "$BUDGETS" \
  --schedules "$SCHEDULES" \
  "$@"
