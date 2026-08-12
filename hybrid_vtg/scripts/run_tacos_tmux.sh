#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
SESSION_NAME="${TACOS_TMUX_SESSION:-tacos-vtg-10}"
ASSET_ROOT="${TACOS_ASSET_ROOT:-$PROJECT_ROOT/assets}"
GPU="${TACOS_GPU:-0}"
SEED="${TACOS_SEED:-42}"
SUBSET="${TACOS_SUBSET:-10}"
LOG_ROOT="${TACOS_LOG_ROOT:-$PROJECT_ROOT/results/logs/tacos-matrix}"
[[ "$ASSET_ROOT" == /* ]] || ASSET_ROOT="$PROJECT_ROOT/$ASSET_ROOT"
[[ "$LOG_ROOT" == /* ]] || LOG_ROOT="$PROJECT_ROOT/$LOG_ROOT"

usage() {
  cat <<'EOF'
Run the nine-experiment TACoS matrix in a detached tmux session.

Usage:
  scripts/run_tacos_tmux.sh

Optional environment variables:
  TACOS_TMUX_SESSION  tmux session name (default: tacos-vtg-10)
  TACOS_ASSET_ROOT    downloaded assets root (default: ./assets)
  TACOS_GPU           CUDA device index (default: 0)
  TACOS_SEED          benchmark seed (default: 42)
  TACOS_SUBSET        query percentage (default: 10)
  TACOS_LOG_ROOT      per-run log directory

Monitor:
  tmux attach -t tacos-vtg-10
  tail -f results/logs/tacos-matrix/matrix.log
EOF
}

run_worker() {
  cd "$PROJECT_ROOT"
  if [[ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    echo "missing virtual environment: $PROJECT_ROOT/.venv" >&2
    echo "create it and run: pip install -e '.[downloads,test]'" >&2
    exit 1
  fi

  # The activation happens inside the detached tmux worker, not only in the
  # SSH shell which launches it.
  source "$PROJECT_ROOT/.venv/bin/activate"
  if ! command -v hybrid-vtg >/dev/null 2>&1; then
    echo "hybrid-vtg is unavailable after activating $PROJECT_ROOT/.venv" >&2
    echo "run: pip install -e '.[downloads,test]'" >&2
    exit 1
  fi
  if ! python -c 'import decord' >/dev/null 2>&1; then
    echo "missing native TimeLens video reader: decord" >&2
    echo "install inside .venv with: pip install 'qwen-vl-utils[decord]>=0.0.14'" >&2
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES="$GPU"
  mkdir -p "$LOG_ROOT"
  exec > >(tee -a "$LOG_ROOT/matrix.log") 2>&1

  echo "started: $(date --iso-8601=seconds)"
  echo "project: $PROJECT_ROOT"
  echo "python: $(command -v python)"
  echo "hybrid-vtg: $(command -v hybrid-vtg)"
  echo "assets: $ASSET_ROOT"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "subset: $SUBSET; seed: $SEED"

  local data="$ASSET_ROOT/datasets/tacos"
  local unitime_adapter="$ASSET_ROOT/checkpoints/unitime/adapter"
  local unitime_base="$ASSET_ROOT/checkpoints/unitime/qwen2-vl-7b"
  local timelens2="$ASSET_ROOT/checkpoints/timelens2-4b"
  local timelens7="$ASSET_ROOT/checkpoints/timelens-7b"
  local -a missing=()
  for path in "$data" "$unitime_adapter" "$unitime_base" "$timelens2" "$timelens7"; do
    [[ -e "$path" ]] || missing+=("$path")
  done
  if ((${#missing[@]})); then
    printf 'missing required asset: %s\n' "${missing[@]}" >&2
    echo "download first with:" >&2
    echo "hybrid-vtg download tacos unitime timelens2-4b timelens-7b --root '$ASSET_ROOT' --accept-licenses --hf-login" >&2
    exit 1
  fi

  local -a common=(
    --benchmark tacos
    --data "$data"
    --subset "$SUBSET"
    --seed "$SEED"
  )
  local -a adaptive=(--method unitime-adaptive --corridor-top-k 4)
  local -a pruned=(
    --encoder-pruning mage
    --encoder-retention 0.5
    --encoder-prune-layer 0
    --post-pruning semvid
    --post-retention 0.125
  )
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

  # Experiment 1: each released model's original/native temporal method.
  run_one unitime-original \
    --model unitime --method unitime-fixed \
    --checkpoint "$unitime_adapter" --base-checkpoint "$unitime_base"
  run_one timelens2-original \
    --model timelens2-4b --method timelens-native --checkpoint "$timelens2"
  run_one timelens7-original \
    --model timelens-7b --method timelens-native --checkpoint "$timelens7"

  # Experiment 2: adaptive top-k=4 temporal routing.
  run_one unitime-adaptive \
    --model unitime "${adaptive[@]}" \
    --checkpoint "$unitime_adapter" --base-checkpoint "$unitime_base"
  run_one timelens2-adaptive \
    --model timelens2-4b "${adaptive[@]}" --checkpoint "$timelens2"
  run_one timelens7-adaptive \
    --model timelens-7b "${adaptive[@]}" --checkpoint "$timelens7"

  # Experiment 3: adaptive routing plus both independent spatial policies.
  run_one unitime-adaptive-mage-semvid \
    --model unitime "${adaptive[@]}" "${pruned[@]}" \
    --checkpoint "$unitime_adapter" --base-checkpoint "$unitime_base"
  run_one timelens2-adaptive-mage-semvid \
    --model timelens2-4b "${adaptive[@]}" "${pruned[@]}" --checkpoint "$timelens2"
  run_one timelens7-adaptive-mage-semvid \
    --model timelens-7b "${adaptive[@]}" "${pruned[@]}" --checkpoint "$timelens7"

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
  "TACOS_TMUX_SESSION=$SESSION_NAME" \
  "TACOS_ASSET_ROOT=$ASSET_ROOT" \
  "TACOS_GPU=$GPU" \
  "TACOS_SEED=$SEED" \
  "TACOS_SUBSET=$SUBSET" \
  "TACOS_LOG_ROOT=$LOG_ROOT" \
  bash "$SCRIPT_PATH" --worker)
tmux new-session -d -s "$SESSION_NAME" "$worker_command"

echo "started detached tmux session: $SESSION_NAME"
echo "attach: tmux attach -t '$SESSION_NAME'"
echo "summary log: $LOG_ROOT/matrix.log"
