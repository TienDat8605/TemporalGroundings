#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_ROOT="$(cd "$EVALUATION_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPOSITORY_ROOT/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLM_VIDEO_DECODE_BACKEND="${VLM_VIDEO_DECODE_BACKEND:-pyav}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/timelens2-matplotlib}"

DATASETS="${VTG_DATASETS:-vue-tr-v2,momentseeker,ego4d-nlq-v2,qvhighlights-timelens}"
PHASE="${VTG_PHASE:-all}"
RUN_NAME="${VTG_RUN_NAME:-smoke}"
MAX_SAMPLES="${VTG_MAX_SAMPLES:-10}"
BUDGETS="${VTG_BUDGETS:-64,128}"
PROMPT_MODES="${VTG_PROMPT_MODES:-controlled,native-style}"
MODEL="${VTG_MODEL:-MCG-NJU/TimeLens2-4B}"
EMBEDDING_MODEL="${VTG_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
DATA_ROOT="${TIMELENS2_DATA_ROOT:-$REPOSITORY_ROOT/data}"
OUTPUT_ROOT="${TIMELENS2_VTG_OUTPUT_ROOT:-$WORKSPACE_ROOT/results/vtg_search}"
CACHE_ROOT="${TIMELENS2_SEARCH_CACHE:-$EVALUATION_ROOT/.cache/vtg-search}"

if [[ "$PHASE" != "validate" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'error: NVIDIA GPU is required for route and ground phases\n' >&2
  exit 2
fi

IFS=',' read -r -a dataset_list <<< "$DATASETS"
mkdir -p "$OUTPUT_ROOT/$RUN_NAME" "$CACHE_ROOT"
cd "$EVALUATION_ROOT"

declare -A requested_datasets=()
for dataset in "${dataset_list[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  [[ -n "$dataset" ]] || continue
  requested_datasets["$dataset"]=1
  case "$dataset" in
    vue-tr-v2)
      dataset_root="${VUE_TR_V2_ROOT:-$DATA_ROOT/VUE_TR_V2}"
      ;;
    momentseeker)
      dataset_root="${MOMENT_SEEKER_ROOT:-$DATA_ROOT/MomentSeeker}"
      ;;
    ego4d-nlq-v2)
      dataset_root="${EGO4D_NLQ_V2_ROOT:-$DATA_ROOT/Ego4D-NLQ-v2}"
      ;;
    qvhighlights-timelens)
      dataset_root="${TIMELENS_BENCH_ROOT:-$DATA_ROOT/TimeLens-Bench}"
      ;;
    *)
      printf 'error: unknown dataset %s\n' "$dataset" >&2
      exit 2
      ;;
  esac
  log_dir="$OUTPUT_ROOT/$RUN_NAME/$dataset"
  mkdir -p "$log_dir"
  python run_vtg_search.py \
    --dataset "$dataset" \
    --phase "$PHASE" \
    --data-root "$dataset_root" \
    --output-root "$OUTPUT_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --run-name "$RUN_NAME" \
    --max-samples "$MAX_SAMPLES" \
    --budgets "$BUDGETS" \
    --prompt-modes "$PROMPT_MODES" \
    --model "$MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    "$@" 2>&1 | tee -a "$log_dir/$PHASE.log"
done

if [[ "$PHASE" == "all" || "$PHASE" == "evaluate" ]]; then
  compare_args=()
  for core_dataset in \
    vue-tr-v2 momentseeker ego4d-nlq-v2 qvhighlights-timelens; do
    if [[ -z "${requested_datasets[$core_dataset]:-}" ]]; then
      compare_args+=(--allow-incomplete)
      break
    fi
  done
  python compare_vtg_search.py "$OUTPUT_ROOT/$RUN_NAME" "${compare_args[@]}"
fi
