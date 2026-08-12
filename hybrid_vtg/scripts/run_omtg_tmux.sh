#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
SESSION_NAME="${OMTG_TMUX_SESSION:-omtg-qwen-scene-10}"
ASSET_ROOT="${OMTG_ASSET_ROOT:-$PROJECT_ROOT/assets}"
GPU="${OMTG_GPU:-0}"
SEED="${OMTG_SEED:-42}"
SUBSET="${OMTG_SUBSET:-10}"
RERUN="${OMTG_RERUN:-0}"
LOG_ROOT="${OMTG_LOG_ROOT:-$PROJECT_ROOT/results/logs/omtg-qwen-scene-window}"
[[ "$ASSET_ROOT" == /* ]] || ASSET_ROOT="$PROJECT_ROOT/$ASSET_ROOT"
[[ "$LOG_ROOT" == /* ]] || LOG_ROOT="$PROJECT_ROOT/$LOG_ROOT"

usage() {
  cat <<'EOF'
Run four matched Qwen3-VL-4B scene-window experiments on OMTG in detached tmux.

Usage:
  scripts/run_omtg_tmux.sh

Optional environment variables:
  OMTG_TMUX_SESSION  tmux session name (default: omtg-qwen-scene-10)
  OMTG_ASSET_ROOT    downloaded assets root (default: ./assets)
  OMTG_GPU           CUDA device index (default: 0)
  OMTG_SEED          benchmark seed (default: 42)
  OMTG_SUBSET        query percentage (default: 10)
  OMTG_RERUN         set to 1 to replace prior predictions (default: 0)
  OMTG_LOG_ROOT      per-run log directory

Session controls:
  tmux attach -t omtg-qwen-scene-10
  tmux detach-client -s omtg-qwen-scene-10
  tmux kill-session -t omtg-qwen-scene-10
EOF
}

run_worker() {
  cd "$PROJECT_ROOT"
  if [[ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    echo "missing virtual environment: $PROJECT_ROOT/.venv" >&2
    echo "create it and run: pip install -e '.[downloads,test]'" >&2
    exit 1
  fi

  source "$PROJECT_ROOT/.venv/bin/activate"
  if ! command -v hybrid-vtg >/dev/null 2>&1; then
    echo "hybrid-vtg is unavailable after activating $PROJECT_ROOT/.venv" >&2
    echo "run: pip install -e '.[downloads,test]'" >&2
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES="$GPU"
  if [[ "$RERUN" != "0" && "$RERUN" != "1" ]]; then
    echo "OMTG_RERUN must be 0 or 1" >&2
    exit 2
  fi
  mkdir -p "$LOG_ROOT"
  exec > >(tee -a "$LOG_ROOT/matrix.log") 2>&1

  echo "started: $(date --iso-8601=seconds)"
  echo "project: $PROJECT_ROOT"
  echo "python: $(command -v python)"
  echo "hybrid-vtg: $(command -v hybrid-vtg)"
  echo "assets: $ASSET_ROOT"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "subset: $SUBSET; seed: $SEED; frame budget: 64"

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
    --model qwen3-vl-4b
    --method coarse-to-fine-64
    --subset "$SUBSET"
    --seed "$SEED"
  )
  [[ "$RERUN" == "0" ]] || common+=(--rerun)
  local failures=0

  run_one() {
    local name="$1"
    shift
    local log="$LOG_ROOT/$name.log"
    echo
    echo "[$(date --iso-8601=seconds)] START $name"
    if hybrid-vtg run "${common[@]}" "$@" 2>&1 | tee "$log"; then
      echo "[$(date --iso-8601=seconds)] PASS  $name"
    else
      local status=${PIPESTATUS[0]}
      echo "[$(date --iso-8601=seconds)] FAIL  $name (exit $status)"
      failures=$((failures + 1))
    fi
  }

  run_one qwen3-vl-4b-dense
  run_one qwen3-vl-4b-mage \
    --encoder-pruning mage --encoder-retention 0.5 --encoder-prune-layer 0
  run_one qwen3-vl-4b-semvid \
    --post-pruning semvid --post-retention 0.125
  run_one qwen3-vl-4b-mage-semvid \
    --encoder-pruning mage --encoder-retention 0.5 --encoder-prune-layer 0 \
    --post-pruning semvid --post-retention 0.125

  echo
  echo "finished: $(date --iso-8601=seconds); failures: $failures"
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
  echo "tmux is not installed" >&2
  exit 1
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

echo "started detached tmux session: $SESSION_NAME"
echo "attach: tmux attach -t '$SESSION_NAME'"
echo "summary log: $LOG_ROOT/matrix.log"
