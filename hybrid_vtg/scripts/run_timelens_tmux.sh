#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
SESSION_NAME="${TIMELENS_TMUX_SESSION:-momentseeker-bench}"
ASSET_ROOT="${TIMELENS_ASSET_ROOT:-$PROJECT_ROOT/assets}"
GPU="${TIMELENS_GPU:-0}"
SEED="${TIMELENS_SEED:-42}"
SUBSET="${TIMELENS_SUBSET:-100}"
MODEL="${TIMELENS_MODEL:-timelens2-4b}"
METHOD="${TIMELENS_METHOD:-anchored-corridor-64}"
BENCHMARK="${TIMELENS_BENCHMARK:-momentseeker}"
RERUN="${TIMELENS_RERUN:-0}"
LOG_ROOT="${TIMELENS_LOG_ROOT:-$PROJECT_ROOT/results/logs/momentseeker}"
[[ "$ASSET_ROOT" == /* ]] || ASSET_ROOT="$PROJECT_ROOT/$ASSET_ROOT"
[[ "$LOG_ROOT" == /* ]] || LOG_ROOT="$PROJECT_ROOT/$LOG_ROOT"

usage() {
  cat <<'EOF'
Run Video Temporal Grounding evaluation in a detached tmux session.

Usage:
  scripts/run_timelens_tmux.sh

Optional environment variables:
  TIMELENS_TMUX_SESSION  tmux session name (default: momentseeker-bench)
  TIMELENS_ASSET_ROOT    downloaded assets root (default: ./assets)
  TIMELENS_GPU           CUDA device index (default: 0)
  TIMELENS_SEED          benchmark seed (default: 42)
  TIMELENS_SUBSET        query percentage (default: 100)
  TIMELENS_MODEL         grounding model (default: timelens2-4b)
  TIMELENS_METHOD        VTG method (default: anchored-corridor-64)
  TIMELENS_BENCHMARK     benchmark split (default: momentseeker)
  TIMELENS_RERUN         set to 1 to replace prior predictions (default: 0)
  TIMELENS_LOG_ROOT      log directory (default: results/logs/momentseeker)

Session controls:
  tmux attach -t momentseeker-bench
  tmux detach-client -s momentseeker-bench
  tmux kill-session -t momentseeker-bench
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
  local subset_tag
  if [[ "$SUBSET" == "100" ]]; then
    subset_tag="p100"
  else
    subset_tag="p0${SUBSET}"
  fi
  local run_log="$LOG_ROOT/${BENCHMARK}--${MODEL}--${METHOD}--${subset_tag}.log"
  exec > >(tee -a "$run_log") 2>&1

  echo "================================================================================"
  echo "TimeLens-Bench Runner"
  echo "started: $(date --iso-8601=seconds)"
  echo "project: $PROJECT_ROOT"
  echo "python: $(command -v python)"
  echo "hybrid-vtg: $(command -v hybrid-vtg)"
  echo "benchmark: $BENCHMARK"
  echo "model: $MODEL"
  echo "method: $METHOD"
  echo "subset: $SUBSET%; seed: $SEED"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "log: $run_log"
  echo "================================================================================"

  local data="$ASSET_ROOT/datasets/$BENCHMARK"
  if [[ ! -d "$data" ]]; then
    echo "missing required dataset: $data" >&2
    echo "download first with:" >&2
    echo "hybrid-vtg download $BENCHMARK --root '$ASSET_ROOT' --accept-licenses" >&2
    exit 1
  fi

  local -a cmd=(
    hybrid-vtg run
    --benchmark "$BENCHMARK"
    --data "$data"
    --model "$MODEL"
    --method "$METHOD"
    --subset "$SUBSET"
    --seed "$SEED"
  )
  [[ "$RERUN" == "0" ]] || cmd+=(--rerun)

  echo
  echo "[$(date --iso-8601=seconds)] START ${cmd[*]}"
  if "${cmd[@]}"; then
    echo "[$(date --iso-8601=seconds)] PASS  TimeLens-Bench evaluation completed successfully."
  else
    local status=$?
    echo "[$(date --iso-8601=seconds)] FAIL  TimeLens-Bench evaluation exited with status $status."
    exit "$status"
  fi
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
  fi
  if [[ "${1:-}" == "--worker" ]]; then
    run_worker
    exit 0
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed; please run: apt install -y tmux" >&2
    exit 1
  fi

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' is already running."
    echo "attach with: tmux attach -t $SESSION_NAME"
    echo "tail logs with: tail -f $LOG_ROOT/${BENCHMARK}--${MODEL}--${METHOD}--p0${SUBSET}.log"
    exit 0
  fi

  mkdir -p "$LOG_ROOT"
  tmux new-session -d -s "$SESSION_NAME" \
    "bash \"$SCRIPT_PATH\" --worker"

  echo "Spawned detached tmux session: $SESSION_NAME"
  echo "Log file: $LOG_ROOT/${BENCHMARK}--${MODEL}--${METHOD}--p0${SUBSET}.log"
  echo
  echo "Commands:"
  echo "  View / Attach:   tmux attach -t $SESSION_NAME"
  echo "  Detach:          Press Ctrl+b then d"
  echo "  Tail log:        tail -f $LOG_ROOT/${BENCHMARK}--${MODEL}--${METHOD}--p0${SUBSET}.log"
  echo "  Kill session:    tmux kill-session -t $SESSION_NAME"
}

main "$@"
