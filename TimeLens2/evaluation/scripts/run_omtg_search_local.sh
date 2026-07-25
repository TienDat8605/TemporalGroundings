#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_ROOT="$(cd "$EVALUATION_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPOSITORY_ROOT/.." && pwd)"

# This is a single-GPU experiment, but the host may have multiple GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TIMELENS2_ATTN_IMPLEMENTATION="${TIMELENS2_ATTN_IMPLEMENTATION:-sdpa}"
export VLM_VIDEO_DECODE_BACKEND="${VLM_VIDEO_DECODE_BACKEND:-pyav}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATA_ROOT="${TIMELENS2_DATA_ROOT:-$REPOSITORY_ROOT/data}"
OMTG_ROOT="${OMTG_BENCH_ROOT:-$DATA_ROOT/OMTGBench}"
OUTPUT_ROOT="${TIMELENS2_OMTG_OUTPUT_ROOT:-$WORKSPACE_ROOT/results/omtg_residual_search}"
CACHE_ROOT="${TIMELENS2_SEARCH_CACHE:-$EVALUATION_ROOT/.cache/omtg-search}"

PHASE="${OMTG_PHASE:-all}"
RUN_NAME="${OMTG_RUN_NAME:-smoke}"
MAX_SAMPLES="${OMTG_MAX_SAMPLES:-25}"
BUDGETS="${OMTG_BUDGETS:-64,128}"
SCHEDULES="${OMTG_SCHEDULES:-score-window-local,residual-window-local,residual-window-local-no-stop}"
MODEL="${OMTG_MODEL:-MCG-NJU/TimeLens2-4B}"
EMBEDDING_MODEL="${OMTG_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"

if [[ "$PHASE" != "validate" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'error: NVIDIA GPU and nvidia-smi are required for routing and grounding\n' >&2
  exit 2
fi

if [[ ! -f "$OMTG_ROOT/OMTGBench.tsv" || ! -d "$OMTG_ROOT/videos" ]]; then
  printf 'error: OMTG Bench is not available at %s\n' "$OMTG_ROOT" >&2
  printf 'download it first with:\n' >&2
  printf '  TIMELENS2_DATA_ROOT=%q bash %q\n' "$DATA_ROOT" "$SCRIPT_DIR/download_omtg_bench.sh" >&2
  exit 2
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR" "$CACHE_ROOT"
LOG_PATH="$RUN_DIR/${PHASE}.log"
printf 'GPU selection: CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Data:          %s\n' "$OMTG_ROOT"
printf 'Output:        %s/%s\n' "$OUTPUT_ROOT" "$RUN_NAME"
printf 'Frame cache:   %s/%s\n' "$CACHE_ROOT" "$RUN_NAME"

cd "$EVALUATION_ROOT"
python run_omtg_search.py \
  --phase "$PHASE" \
  --data-root "$OMTG_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --run-name "$RUN_NAME" \
  --max-samples "$MAX_SAMPLES" \
  --budgets "$BUDGETS" \
  --schedules "$SCHEDULES" \
  --model "$MODEL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --attention "$TIMELENS2_ATTN_IMPLEMENTATION" \
  "$@" 2>&1 | tee -a "$LOG_PATH"
