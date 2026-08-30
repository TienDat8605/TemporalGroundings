#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
SESSION_NAME="${OMTG_TMUX_SESSION:-omtg-qwen3-8b-matrix}"
ASSET_ROOT="${OMTG_ASSET_ROOT:-$PROJECT_ROOT/assets}"
GPU="${OMTG_GPU:-0}"
SEED="${OMTG_SEED:-42}"
SUBSET="${OMTG_SUBSET:-100}"
RERUN="${OMTG_RERUN:-0}"
LOG_ROOT="${OMTG_LOG_ROOT:-$PROJECT_ROOT/results/logs/omtg-qwen3-8b}"
MODEL="qwen3-vl-8b"
[[ "$ASSET_ROOT" == /* ]] || ASSET_ROOT="$PROJECT_ROOT/$ASSET_ROOT"
[[ "$LOG_ROOT" == /* ]] || LOG_ROOT="$PROJECT_ROOT/$LOG_ROOT"

usage() {
  cat <<'USAGE'
Run base Qwen3-VL-8B experiments on OMTG in detached tmux (Native 2FPS, SGDE-64, SGDE-128, SGDE-256).

Usage:
  scripts/run_qwen3_8b_omtg_matrix.sh

Optional environment variables:
  OMTG_TMUX_SESSION  tmux session name (default: omtg-qwen3-8b-matrix)
  OMTG_ASSET_ROOT    downloaded assets root (default: ./assets)
  OMTG_GPU           CUDA device index (default: 0)
  OMTG_SEED          benchmark seed (default: 42)
  OMTG_SUBSET        query percentage (default: 100)
  OMTG_RERUN         set to 1 to replace prior predictions (default: 0)
  OMTG_LOG_ROOT      per-run log directory

Session controls:
  tmux attach -t omtg-qwen3-8b-matrix
  tmux detach-client -s omtg-qwen3-8b-matrix
  tmux kill-session -t omtg-qwen3-8b-matrix
USAGE
}

run_worker() {
  cd "$PROJECT_ROOT"
  if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
  fi

  if ! command -v hybrid-vtg >/dev/null 2>&1; then
    echo "hybrid-vtg is unavailable. Run: pip install -e '.[downloads,test]'" >&2
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES="$GPU"
  export PYTHONUNBUFFERED=1
  if [[ "$RERUN" != "0" && "$RERUN" != "1" ]]; then
    echo "OMTG_RERUN must be 0 or 1" >&2
    exit 2
  fi
  mkdir -p "$LOG_ROOT"
  exec > >(tee -a "$LOG_ROOT/matrix.log") 2>&1

  echo "================================================================================"
  echo "OMTG Benchmark Matrix: Base Qwen3-VL-8B"
  echo "started: $(date --iso-8601=seconds)"
  echo "project: $PROJECT_ROOT"
  echo "python: $(command -v python)"
  echo "hybrid-vtg: $(command -v hybrid-vtg)"
  echo "assets: $ASSET_ROOT"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "model: $MODEL (Qwen/Qwen3-VL-8B-Instruct)"
  echo "subset: $SUBSET%; seed: $SEED"
  echo "methods: native, sgde-64, sgde-128, sgde-256"
  echo "================================================================================"

  local data="$ASSET_ROOT/datasets/omtg"
  if [[ ! -d "$data" ]]; then
    echo "missing required dataset: $data" >&2
    echo "download first with:" >&2
    echo "hybrid-vtg download omtg --root '$ASSET_ROOT' --accept-licenses --hf-login" >&2
    exit 1
  fi

  local -a common=(
    --benchmark omtg
    --data "$data"
    --model "$MODEL"
    --subset "$SUBSET"
    --seed "$SEED"
  )
  [[ "$RERUN" == "0" ]] || common+=(--rerun)
  local failures=0

  run_one() {
    local method_name="$1"
    local run_tag="$2"
    local log="$LOG_ROOT/$run_tag.log"
    echo
    echo "================================================================================"
    echo "[$(date --iso-8601=seconds)] START $run_tag (method: $method_name)"
    echo "================================================================================"
    if hybrid-vtg run "${common[@]}" --method "$method_name" 2>&1 | tee "$log"; then
      echo "[$(date --iso-8601=seconds)] PASS  $run_tag"
    else
      local status=${PIPESTATUS[0]}
      echo "[$(date --iso-8601=seconds)] FAIL  $run_tag (exit $status)"
      failures=$((failures + 1))
    fi
  }

  # 1. Native 2FPS
  run_one native qwen3-vl-8b-native-2fps

  # 2. Adaptive SGDE at 64 frames
  run_one sgde-64 qwen3-vl-8b-sgde-64f

  # 3. Adaptive SGDE at 128 frames
  run_one sgde-128 qwen3-vl-8b-sgde-128f

  # 4. Adaptive SGDE at 256 frames
  run_one sgde-256 qwen3-vl-8b-sgde-256f

  echo
  echo "================================================================================"
  echo "All experiments finished: $(date --iso-8601=seconds); total failures: $failures"
  echo "================================================================================"
  exit "$failures"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
fi
if [[ $# -gt 0 ]]; then
  usage >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed; running directly..."
  run_worker
  exit 0
fi

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "attach with: tmux attach -t '$SESSION_NAME'" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
worker_command=$(printf '%q ' \
  env \
  "OMTG_TMUX_SESSION=$SESSION_NAME" \
  "OMTG_ASSET_ROOT=$ASSET_ROOT" \
  "OMTG_GPU=$GPU" \
  "OMTG_SEED=$SEED" \
  "OMTG_SUBSET=$SUBSET" \
  "OMTG_RERUN=$RERUN" \
  "OMTG_LOG_ROOT=$LOG_ROOT" \
  bash "$SCRIPT_PATH" --worker)
tmux new-session -d -s "$SESSION_NAME" "$worker_command"

echo "Started detached tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t '$SESSION_NAME'"
echo "Summary log: $LOG_ROOT/matrix.log"
