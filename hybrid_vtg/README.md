# Hybrid VTG

Hybrid VTG is a test-only runner for frozen-model video temporal grounding. This branch intentionally contains two inference methods:

- `coarse-to-fine-64`: scene-window routing and local grounding under a strict 64 source-frame budget.
- `native`: checkpoint-native inference for UniTime, TimeLens, and TimeLens2.

No training or fine-tuning is performed. The reusable Qwen backends still support independent Mage encoder pruning and SemVID post-encoder pruning.

## Scene-window coarse-to-fine grounding

`coarse-to-fine-64` uses a deliberately small pipeline:

1. PySceneDetect divides a long video into 20–60 second windows. A deterministic uniform-window fallback is used when scene detection finds no boundaries.
2. Frozen `Qwen/Qwen3-VL-Embedding-2B` on CPU ranks every window against the query.
3. The router and selected local windows share exactly 64 decoded source frames.
4. The frozen grounder predicts each selected window independently.
5. Local timestamps are mapped to absolute video time.
6. Cross-window duplicates are fused; all final occurrences are returned chronologically.

A video with only one window bypasses routing and gives all 64 frames to the grounder.

PySceneDetect results are cached once per video revision and detector policy under `results/cache/methods/coarse-to-fine-64/scenes/`. The cache is shared by all queries, seeds, pruning variants, and reruns. Changing the video file, its recorded duration, or the detector policy produces a new cache entry.

The embedding router uses the same shared-cache principle under `results/cache/methods/coarse-to-fine-64/router/`. Normalized video-window embeddings are keyed by the video revision, routed boundaries, exact sampled timestamps, router model, and embedding-policy version. The router passes pre-decoded JPEG lists with processor-side frame sampling disabled, preserving the method's exact frame budget. Query embeddings are cached separately by exact query text. A new query therefore reuses the expensive video embeddings and performs only text encoding plus a local dot product; matched pruning runs normally reuse both sides.

### Cross-window fusion

Fusion is provenance-aware. Candidates are seeded in descending router-score order. A seed accepts at most one prediction from each other window, and only when temporal IoU is strictly greater than `0.6`. Predictions from the same local window are never fused. Adjacent and non-overlapping spans are not merged.

For a fusion group, start and end boundaries are averaged with softmax-normalized router scores. A higher-ranked window therefore has more influence:

```text
window 2, router 0.90: [31.0, 39.0] --+
window 3, router 0.70: [32.0, 40.0] --+--> [31.45, 39.45]

window 2: [46.0, 49.0] -------------------> unchanged
```

Prediction telemetry records every routed window's range and score, frame allocation, dense and retained encoder tokens, SemVID input and output visual tokens, LLM input/output tokens, local encode/predict timing, local and absolute spans, and final fusion membership.

## Pruning

Both pruning policies are optional and disabled by default. They do not change scene routing or the 64-frame allocation.

- Mage removes complete Qwen merger cells before a configurable vision-transformer block. The matched experiment retains `0.5` at layer `0`.
- SemVID selects query-, context-, and motion-aware evidence immediately before each local LLM prefill. The matched experiment retains `0.125` relative to dense local evidence.

Mage only:

```bash
--encoder-pruning mage --encoder-retention 0.5 --encoder-prune-layer 0
```

SemVID only:

```bash
--post-pruning semvid --post-retention 0.125
```

Use both blocks for the combined configuration. When both are enabled, post retention cannot exceed encoder retention.

## Install

Python 3.10 or newer is required. Install a PyTorch build compatible with the machine's CUDA driver, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[downloads,test]'
```

The base dependencies include headless OpenCV, PySceneDetect, Transformers, Sentence Transformers, and the Qwen video utilities. CUDA is strongly recommended for the generative grounder; the embedding router intentionally stays on CPU.

## Run one experiment

Download OMTG:

```bash
hybrid-vtg download omtg --root ./assets --accept-licenses --hf-login
```

The Qwen checkpoints are fetched into the Hugging Face cache on first use. Run the dense scene-window baseline on a reproducible 10% query subset:

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model qwen3-vl-4b \
  --method coarse-to-fine-64 \
  --subset 10 \
  --seed 42
```

Append either or both pruning blocks shown above. Runs resume by sample ID. Add `--rerun` to replace prior predictions for that configuration.

Pruning settings are encoded in distinct result directories:

```text
results/runs/omtg/qwen3-vl-4b/coarse-to-fine-64/seed-42/
results/runs/omtg/qwen3-vl-4b--enc-mage-r0.5-l0/coarse-to-fine-64/seed-42/
results/runs/omtg/qwen3-vl-4b--post-semvid-r0.125/coarse-to-fine-64/seed-42/
results/runs/omtg/qwen3-vl-4b--enc-mage-r0.5-l0--post-semvid-r0.125/coarse-to-fine-64/seed-42/
```

## Run the four matched OMTG experiments in tmux

The runner executes dense, Mage, SemVID, and Mage + SemVID sequentially with the same OMTG subset, seed, scene-window policy, embedding router, Qwen3-VL-4B grounder, and 64-frame budget:

```bash
scripts/run_omtg_tmux.sh
```

Choose the subset and GPU at launch:

```bash
OMTG_SUBSET=100 OMTG_GPU=1 scripts/run_omtg_tmux.sh
```

Replace matching prior results after a code change:

```bash
OMTG_RERUN=1 scripts/run_omtg_tmux.sh
```

Attach, detach, and stop the default session:

```bash
tmux attach -t omtg-qwen-scene-10
tmux detach-client -s omtg-qwen-scene-10
tmux kill-session -t omtg-qwen-scene-10
```

Inside tmux, `Ctrl-b d` also detaches without stopping the worker. Follow the aggregate log with:

```bash
tail -f results/logs/omtg-qwen-scene-window/matrix.log
```

The runner accepts `OMTG_TMUX_SESSION`, `OMTG_ASSET_ROOT`, `OMTG_GPU`, `OMTG_SEED`, `OMTG_SUBSET`, `OMTG_RERUN`, and `OMTG_LOG_ROOT` overrides.

## Native model controls

`native` dispatches by model family. UniTime uses its fixed 32-second coarse retrieval followed by local fine grounding (or a direct fine pass for videos up to 64 seconds). TimeLens-7B, TimeLens-8B, and TimeLens2-4B use their released whole-video 2 FPS prompts. UniTime's encoded-evidence path supports Mage and SemVID; native TimeLens inference is intentionally dense and rejects pruning flags.

```bash
hybrid-vtg download tacos timelens2-4b \
  --root ./assets --accept-licenses --hf-login

hybrid-vtg run \
  --benchmark tacos \
  --data ./assets/datasets/tacos \
  --model timelens2-4b \
  --checkpoint ./assets/checkpoints/timelens2-4b \
  --method native \
  --subset 10 \
  --seed 42
```

## Benchmarks and outputs

The benchmark registry contains OMTG, TACoS, and QVHighlights. `--subset` accepts percentages from `0` through `100`; sampling is deterministic over query IDs for a given seed, and smaller percentages are prefixes of larger ones.

Each run writes a manifest, resumable `predictions.jsonl`, errors, and per-subset metrics under `results/runs/`. OMTG is evaluated as multi-interval grounding. TACoS reports single-result moment-retrieval metrics against its references. QVHighlights test labels are hidden, so complete successful runs export a submission instead of local metrics.

Run the checks with:

```bash
PYTHONPATH=src pytest
ruff check .
bash -n scripts/run_omtg_tmux.sh
```
