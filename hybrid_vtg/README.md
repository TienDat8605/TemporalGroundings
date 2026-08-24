# Hybrid VTG

Hybrid VTG is a test-only runner for frozen-model video temporal grounding. This branch contains four inference methods:

- `sgde-64`: scout-guided dense evidence grounding (Idea 3) separating cheap global scouting from dense anchored local LVLM verification under a 64-frame budget.
- `anchored-corridor-64`: multi-view semantic routing, safe full-video fallback, and one globally anchored grounding call with exactly 64 evidence frames.
- `coarse-to-fine-64`: scene-window routing and local grounding under a strict 64 source-frame budget.
- `native`: checkpoint-native inference for UniTime, TimeLens, and TimeLens2.

No training or fine-tuning is performed. The reusable Qwen backends still support independent Mage encoder pruning and SemVID post-encoder pruning.

## Adaptive Scout-Guided Dense Evidence Grounding (SGDE-64)

`sgde-64` implements the two-stage **Adaptive Scout-Guided Dense Evidence Grounding (Idea 3)** paradigm:

1. **Stage 1 (Scout Timeline & Candidate Proposal Mining)**:
   - Computes or reuses cached 1 FPS visual scout embeddings (e.g. `google/siglip2-base-patch16-224`, `nvidia/llama-nemotron-embed-vl-1b-v2`, or `Qwen/Qwen3-VL-Embedding-2B`).
   - Normalizes timeline with robust median and MAD: $z(t) = \frac{s(t) - \operatorname{median}(s)}{\operatorname{MAD}(s) + \epsilon}$ with conservative temporal smoothing.
   - Extracts candidate proposals using hysteresis connected components, penalized interval scoring $J(a,b)$, and multi-scale density windows (8s, 16s, 32s).
   - Applies 1D temporal NMS to retain diverse high-confidence candidate intervals.

2. **Stage 2 (Adaptive Anchored Dense Evidence & Multi-Interval Verification)**:
   - Plans adaptive evidence corridor: protects global timeline anchors across the video while concentrating dense frames inside candidate interiors, pre/post context padding, and boundary transitions.
   - Dynamically scales budget (64f base, up to 128f fallback for complex or low-confidence samples).
   - Encodes chronological evidence plan once and performs a single-call LVLM temporal verification and grounding.
   - Evaluates multi-interval occurrences seamlessly on multi-span benchmarks like OMTG.

### Run Adaptive SGDE on OMTG

Run OMTG benchmark with `timelens2-4b`:

```bash
# Fast 10% diagnostic subset
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model timelens2-4b \
  --method sgde-64 \
  --subset 10 \
  --seed 42

# Full 100% OMTG benchmark evaluation
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model timelens2-4b \
  --method sgde-64 \
  --subset 100 \
  --seed 42
```

### Run Adaptive SGDE on OMTG in detached tmux

Run in the background on GPU (e.g., RTX 3060 12GB):

```bash
TIMELENS_BENCHMARK=omtg \
TIMELENS_MODEL=timelens2-4b \
TIMELENS_METHOD=sgde-64 \
TIMELENS_SUBSET=100 \
TIMELENS_GPU=0 \
scripts/run_timelens_tmux.sh
```

Monitor logs and session:

```bash
# Attach to tmux session
tmux attach -t omtg-timelens2-4b-sgde-64-100

# Tail live output
tail -f results/logs/omtg/omtg--timelens2-4b--sgde-64--p100.log
```

### Run Adaptive SGDE on QVHighlights-TimeLens

```bash
hybrid-vtg run \
  --benchmark qvhighlights-timelens \
  --data ./assets/datasets/qvhighlights-timelens \
  --model timelens2-4b \
  --method sgde-64 \
  --subset 10 \
  --seed 42
```

## One-call anchored corridor grounding

`anchored-corridor-64` is the one-call successor experiment to the independent-window baseline:

1. Reuse cached content-aware windows and the Qwen3-VL-Embedding-2B visual index.
2. Rank with raw, coarse-intent, action-sequence, and object/motion-detail query views.
3. Accept hard routing only when query views and visual views agree and the robust score margin is at least `0.5`; otherwise fail open to uniform full-video evidence.
4. Protect one global anchor for every routed macro-window and spend the rest of exactly 64 grounding frames inside the selected corridor(s).
5. Encode the chronological evidence plan once and make one primary grounding call in original-video time.

The router index has a separate cold/cache ledger and is not hidden inside the 64-frame grounding budget. This first implementation requires dense evidence; Mage and SemVID remain separate follow-up ablations until anchor-aware pruning is validated.

Run the full OMTG experiment with the stronger `tmp3` backend:

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model timelens2-4b \
  --method anchored-corridor-64 \
  --subset 100 \
  --seed 42
```

## Scene-window coarse-to-fine grounding

`coarse-to-fine-64` uses a deliberately small pipeline:

1. PySceneDetect divides a long video into 20–60 second windows. A deterministic uniform-window fallback is used when scene detection finds no boundaries.
2. Frozen `Qwen/Qwen3-VL-Embedding-2B` on CPU ranks every window against the query. It combines a whole-window similarity with a brief-occurrence score from the exact sampled frames, then applies a mild temporal-diversity term when choosing near-tied windows.
3. The router and selected local windows share exactly 64 decoded source frames.
4. The frozen grounder predicts each selected window independently.
5. Local timestamps are mapped to absolute video time.
6. Cross-window duplicates are fused; all final occurrences are returned chronologically.

A video with only one window bypasses routing and gives all 64 frames to the grounder.

PySceneDetect results are cached once per video revision and detector policy under `results/cache/methods/coarse-to-fine-64/scenes/`. The cache is shared by all queries, seeds, pruning variants, and reruns. Changing the video file, its recorded duration, or the detector policy produces a new cache entry.

The embedding router uses the same shared-cache principle under `results/cache/methods/coarse-to-fine-64/router/`. It loads Qwen's embedding-specific `Qwen3VLForEmbedding` implementation at a pinned revision and rejects missing or mismatched checkpoint weights. Before the grounder is loaded, all missing query and visual embeddings are computed on GPU when CUDA is available and cached; the router is then unloaded and its GPU memory released. Normalized whole-window and sampled-frame embeddings are keyed by the video revision, routed boundaries, exact sampled timestamps, router model revision, and embedding-policy version. The exact pre-decoded JPEG list is reused for both views, so the richer scoring does not increase the 64-source-frame budget. A new query therefore reuses the expensive visual embeddings; matched model and pruning runs normally reuse both sides. The cache policy changed with this router upgrade, so vectors produced by the old generic loader are never reused.

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

The base dependencies include headless OpenCV, PySceneDetect, Transformers, Sentence Transformers, and the Qwen video utilities. CUDA is strongly recommended for the generative grounder; the embedding router intentionally stays on CPU.

## Precompute QVHighlights-TimeLens scout features

The resumable preparation script downloads the 1,511-video TimeLens-Bench QVHighlights
release with `aria2c`, samples frames at 1 FPS, and embeds frames and all 1,541 queries
with the pinned `nvidia/llama-nemotron-embed-vl-1b-v2` revision:

```bash
scripts/prepare_qvhighlights_nemotron.sh
```

The model uses one 512-pixel image tile per video frame by default so the base FP16
checkpoint can run on a 4 GB GPU. Outputs are normalized float16 vectors compatible
with the existing scout artifact layout:

```text
assets/features/scouts/nvidia--llama-nemotron-embed-vl-1b-v2/qvhighlights-timelens/
  manifest.json
  queries.npz
  video_embeddings/<video-id>.npz
assets/features/scouts/scout_nvidia--llama-nemotron-embed-vl-1b-v2_qvhighlights-timelens.tar.gz
```

Each video NPZ is written atomically and validated before reuse, so rerunning the script
resumes at the first missing video. Override extraction settings by appending arguments
such as `--device cuda:1`, `--fps 0.5`, or `--batch-size 2`.

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

Each configuration gets one flat run directory. The short suffix is a stable
configuration ID; the readable hyperparameters are stored in every metrics JSON:

```text
results/runs/omtg--qwen3-vl-4b--coarse-to-fine-64--seed-42--a1b2c3d4e5/
  manifest.json
  predictions.jsonl
  errors.jsonl
  metrics-p010.json
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

Each run writes its manifest, resumable `predictions.jsonl`, errors, per-subset metrics,
and (when applicable) submission directly in one directory under `results/runs/`.
Metrics JSON files include the complete run hyperparameters. OMTG is evaluated as
multi-interval grounding. TACoS reports single-result moment-retrieval metrics against
its references. QVHighlights test labels are hidden, so complete successful runs export
a submission instead of local metrics.

Run the checks with:

```bash
PYTHONPATH=src pytest
ruff check .
bash -n scripts/run_omtg_tmux.sh
```
