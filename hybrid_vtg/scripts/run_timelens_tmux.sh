#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
BENCHMARK="${TIMELENS_BENCHMARK:-omtg}"
MODEL="${TIMELENS_MODEL:-timelens2-4b}"
METHOD="${TIMELENS_METHOD:-sgde-64}"
SUBSET="${TIMELENS_SUBSET:-100}"
SEED="${TIMELENS_SEED:-42}"
GPU="${TIMELENS_GPU:-0}"
RERUN="${TIMELENS_RERUN:-0}"
SCOUT_MODEL="${TIMELENS_SCOUT_MODEL:-}"
ASSET_ROOT="${TIMELENS_ASSET_ROOT:-$PROJECT_ROOT/assets}"
SESSION_NAME="${TIMELENS_TMUX_SESSION:-${BENCHMARK}-${MODEL}-${METHOD}-${SUBSET}}"
LOG_ROOT="${TIMELENS_LOG_ROOT:-$PROJECT_ROOT/results/logs/${BENCHMARK}}"
[[ "$ASSET_ROOT" == /* ]] || ASSET_ROOT="$PROJECT_ROOT/$ASSET_ROOT"
[[ "$LOG_ROOT" == /* ]] || LOG_ROOT="$PROJECT_ROOT/$LOG_ROOT"

usage() {
  cat <<'EOF'
Run Video Temporal Grounding evaluation in a detached tmux session.

Usage:
  scripts/run_timelens_tmux.sh

Optional environment variables:
  TIMELENS_BENCHMARK     benchmark split (default: omtg; options: omtg, qvhighlights-timelens, momentseeker, tacos)
  TIMELENS_MODEL         grounding model (default: timelens2-4b; options: timelens2-4b, qwen3-vl-4b)
  TIMELENS_METHOD        VTG method (default: sgde-64; options: sgde-64, anchored-corridor-64, coarse-to-fine-64, native)
  TIMELENS_SUBSET        query percentage (default: 100)
  TIMELENS_SEED          benchmark seed (default: 42)
  TIMELENS_GPU           CUDA device index (default: 0)
  TIMELENS_RERUN         set to 1 to replace prior predictions (default: 0)
  TIMELENS_ASSET_ROOT    downloaded assets root (default: ./assets)
  TIMELENS_TMUX_SESSION  tmux session name
  TIMELENS_LOG_ROOT      log directory (default: results/logs/<benchmark>)

Session controls:
  tmux attach -t <session_name>
  tmux detach-client -s <session_name>
  tmux kill-session -t <session_name>
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

  local subset_tag
  if [[ "$SUBSET" == "100" ]]; then
    subset_tag="p100"
  else
    subset_tag="p0${SUBSET}"
  fi
  local log_file="$LOG_ROOT/${BENCHMARK}--${MODEL}--${METHOD}--${subset_tag}.log"

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' is already running."
    echo "attach with: tmux attach -t $SESSION_NAME"
    echo "tail logs with: tail -f $log_file"
    exit 0
  fi

  mkdir -p "$LOG_ROOT"
  tmux new-session -d -s "$SESSION_NAME" \
    "TIMELENS_SCOUT_MODEL=\"$SCOUT_MODEL\" bash \"$SCRIPT_PATH\" --worker"

  echo "Spawned detached tmux session: $SESSION_NAME"
  echo "Log file: $log_file"
  echo
  echo "Commands:"
  echo "  View / Attach:   tmux attach -t $SESSION_NAME"
  echo "  Detach:          Press Ctrl+b then d"
  echo "  Tail log:        tail -f $log_file"
  echo "  Kill session:    tmux kill-session -t $SESSION_NAME"
}

main "$@"
