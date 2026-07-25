#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_ROOT="$(cd "$EVALUATION_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPOSITORY_ROOT/.." && pwd)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'error: NVIDIA GPU and nvidia-smi are required for TimeLens2 inference\n' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TIMELENS2_ATTN_IMPLEMENTATION="${TIMELENS2_ATTN_IMPLEMENTATION:-sdpa}"
export VLM_VIDEO_DECODE_BACKEND="${VLM_VIDEO_DECODE_BACKEND:-pyav}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTHON_BIN="${PYTHON:-python}"

DATA_ROOT="${TIMELENS2_DATA_ROOT:-$REPOSITORY_ROOT/data}"
OMTG_ROOT="${OMTG_BENCH_ROOT:-$DATA_ROOT/OMTGBench}"
OUTPUT_ROOT="${TIMELENS2_OMTG_BASELINE_OUTPUT_ROOT:-$WORKSPACE_ROOT/results/omtg_2fps}"
RUN_NAME="${OMTG_RUN_NAME:-timelens2-4b-paper-2fps}"
MODEL="${OMTG_MODEL:-MCG-NJU/TimeLens2-4B}"
PROMPT_MODES="${OMTG_PROMPT_MODES:-official,controlled}"
MAX_SAMPLES="${OMTG_MAX_SAMPLES:-0}"
PHASE="${OMTG_PHASE:-all}"

if [[ ! -f "$OMTG_ROOT/OMTGBench.tsv" || ! -d "$OMTG_ROOT/videos" ]]; then
  printf 'error: OMTG Bench is not available at %s\n' "$OMTG_ROOT" >&2
  printf 'download it first with:\n' >&2
  printf '  TIMELENS2_DATA_ROOT=%q bash %q\n' "$DATA_ROOT" "$SCRIPT_DIR/download_omtg_bench.sh" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
"$PYTHON_BIN" - <<'PY'
import sys

try:
    import decord  # noqa: F401
except ImportError:
    raise SystemExit(
        "error: decord is required by qwen-vl-utils; install it with "
        "`python -m pip install decord==0.6.0`"
    )

try:
    import torch
except ImportError:
    raise SystemExit("error: PyTorch is not installed in the active environment")

if not torch.cuda.is_available():
    detail = ""
    try:
        torch.cuda.init()
    except Exception as exc:
        detail = f"\nCUDA initialization detail: {exc}"
    raise SystemExit(
        "error: CUDA is unavailable to PyTorch. Install a PyTorch CUDA build "
        "compatible with the host NVIDIA driver." + detail
    )
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
PY

printf 'GPU selection: CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Data:          %s\n' "$OMTG_ROOT"
printf 'Model:         %s\n' "$MODEL"
printf 'Prompt modes:  %s\n' "$PROMPT_MODES"
printf 'Output:        %s/%s\n' "$OUTPUT_ROOT" "$RUN_NAME"

cd "$EVALUATION_ROOT"
"$PYTHON_BIN" run_omtg_2fps_baseline.py \
  --phase "$PHASE" \
  --data-root "$OMTG_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --model "$MODEL" \
  --prompt-modes "$PROMPT_MODES" \
  --max-samples "$MAX_SAMPLES" \
  --attention "$TIMELENS2_ATTN_IMPLEMENTATION" \
  "$@"
