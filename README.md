# Temporal Groundings: Hybrid-VTG

Temporal Groundings is a research framework for training-free, test-time video temporal grounding (VTG) using frozen Large Vision-Language Models (LVLMs) such as TimeLens2-4B and Qwen3-VL-4B.

---

## Key Inference Paradigm: Adaptive SGDE (SGDE-64)

**Adaptive Scout-Guided Dense Evidence Grounding (Adaptive SGDE / Idea 3)** decouples video temporal grounding into two specialized stages:

1. **Stage 1 (Scout Timeline & Multi-Scale Candidate Mining)**:
   - Evaluates fast 1 FPS visual scout features (e.g., SigLIP-2, Llama-Nemotron-Embed-VL-1B, Qwen3-VL-Embedding-2B).
   - Normalizes timeline using robust median / MAD scaling with temporal smoothing.
   - Extracts candidate moment proposals using hysteresis connected components and multi-scale temporal windows (8s, 16s, 32s).
2. **Stage 2 (Adaptive Anchored Dense Evidence & Verification)**:
   - Dynamically allocates evidence frames: protects global anchors across the full video and concentrates dense frames inside high-value candidate regions and context boundaries.
   - Executes a single-call LVLM inference to ground target occurrences in original-video time.

---

## Quickstart: Running OMTG Benchmark with Adaptive SGDE

### 1. Environment Setup

```bash
cd hybrid_vtg
python -m venv .venv
source .venv/bin/activate
pip install -e '.[downloads,test]'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Download OMTG Benchmark Dataset

```bash
hybrid-vtg download omtg --root ./assets --accept-licenses --hf-login
```

### 3. Run Benchmark

#### Interactive Run
```bash
# 10% diagnostic subset
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model timelens2-4b \
  --method sgde-64 \
  --subset 10 \
  --seed 42

# Full 100% split
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model timelens2-4b \
  --method sgde-64 \
  --subset 100 \
  --seed 42
```

#### Detached Background Run (tmux)
```bash
TIMELENS_BENCHMARK=omtg \
TIMELENS_MODEL=timelens2-4b \
TIMELENS_METHOD=sgde-64 \
TIMELENS_SUBSET=100 \
TIMELENS_GPU=0 \
scripts/run_timelens_tmux.sh
```

Monitor execution:
```bash
tmux attach -t omtg-timelens2-4b-sgde-64-100
tail -f results/logs/omtg/omtg--timelens2-4b--sgde-64--p100.log
```

---

## Repository Structure

- [`hybrid_vtg/`](./hybrid_vtg/): Main Python package and benchmark runner CLI (`hybrid-vtg`).
- [`hybrid_vtg/docs/research-notes/`](./hybrid_vtg/docs/research-notes/): Research notes, mathematical formulations, and experimental telemetry reports.
- [`hybrid_vtg/scripts/`](./hybrid_vtg/scripts/): Automated tmux runners and scout precomputation scripts.
