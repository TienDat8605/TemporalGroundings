#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo "========================================================="
echo "=== STARTING OMTG BENCHMARK & ABLATION CAMPAIGN ==="
echo "=== Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown') ==="
echo "========================================================="

# 1. Baseline: Naive Uniform Downsampling (128f)
echo ""
echo ">>> [1/7] Running Baseline: Naive Uniform Downsampling (128f)..."
python -m hybrid_vtg.cli eval --benchmark omtg --model timelens-8b --method native-128f --subset 100

# 2. Baseline: Random Temporal Crop (128f)
echo ""
echo ">>> [2/7] Running Baseline: Random Temporal Crop (128f)..."
python -m hybrid_vtg.cli eval --benchmark omtg --model timelens-8b --method random-crop-128f --subset 100

# 3. Baseline: Oracle Ground-Truth Corridor (128f)
echo ""
echo ">>> [3/7] Running Baseline: Oracle Ground-Truth Corridor (128f)..."
python -m hybrid_vtg.cli eval --benchmark omtg --model timelens-8b --method oracle-corridor-128f --subset 100

# 4. Ablation Step 1: Raw Cosine (128f)
echo ""
echo ">>> [4/7] Running Ablation Step 1: Raw Cosine Similarity..."
python -m hybrid_vtg.cli eval --benchmark omtg --model timelens-8b --method sgde-step1-raw-cosine --subset 100

# 5. Ablation Step 2: + Z-Score Normalization (128f)
echo ""
echo ">>> [5/7] Running Ablation Step 2: + Z-Score Normalization..."
python -m hybrid_vtg.cli eval --benchmark omtg --model timelens-8b --method sgde-step2-zscore --subset 100

# 6. Ablation Step 3: + Hysteresis & Energy Mining (128f)
echo ""
echo ">>> [6/7] Running Ablation Step 3: + Hysteresis & Energy Mining..."
python -m hybrid_vtg.cli eval --benchmark omtg --model timelens-8b --method sgde-step3-energy --subset 100

# 7. Ablation Step 4: + Duration-Adaptive Geometry (128f, No Merge)
echo ""
echo ">>> [7/7] Running Ablation Step 4: + Duration-Adaptive Geometry..."
python -m hybrid_vtg.cli eval --benchmark omtg --model timelens-8b --method sgde-step4-adaptive-geom --subset 100

echo ""
echo "========================================================="
echo "=== CAMPAIGN COMPLETE! Aggregating metrics... ==="
echo "========================================================="
python scripts/compute_omtg_stratified_metrics.py || true
python scripts/benchmark_latency_and_vram.py || true
echo "All results successfully computed and logged."
