# Hybrid-VTG: training-free temporal routing + SemVID

This package is the primary implementation of the proposed hierarchical VTG method. It replaces TimeLens2 as the research backbone and composes two frozen systems:

1. a cheap SigLIP2 whole-video scan finds high-recall temporal components;
2. the official SemVID Qwen3-VL implementation performs real object/motion/context token pruning and grounding only inside the best component;
3. SigLIP2 resamples a small neighborhood at high FPS and adjusts the predicted boundaries from semantic change and visual continuity.

No training code, optimizer, loss, adapter, checkpoint update, or benchmark-label access exists in the inference path. All models run in evaluation and inference mode.

## What is new relative to SemVID

SemVID sparsifies tokens after frames have reached the expert vision encoder. This package adds query-guided temporal routing before that expensive encoding. The research claim is therefore hierarchical conditional computation:

```text
whole video --cheap frozen scan--> retained temporal components
retained component --SemVID expert encoding + token pruning--> coarse interval
boundary neighborhoods --dense cheap scan--> refined global timestamps
```

SemVID remains an unmodified Git submodule pinned at `432a76928817cdfba7d04c460ac475482cd7c3a4`. TimeLens2 remains in the repository only as a legacy comparison baseline.

## Local GPU setup

Clone with submodules and create a clean environment:

```bash
git clone --recurse-submodules <repository-url>
cd read_papers
conda create -n hybrid-vtg python=3.11 -y
conda activate hybrid-vtg
pip install -r hybrid_vtg/requirements.txt
pip install -e 'hybrid_vtg[test]'
hybrid-vtg doctor
```

If the repository was already cloned:

```bash
git submodule update --init --recursive
```

The default is `Qwen/Qwen3-VL-4B-Thinking` with SDPA, so FlashAttention is optional. A recent CUDA GPU with 24 GiB VRAM is the practical minimum recommendation; an L40/L40S, A100, or H100 gives more headroom. SemVID uses `device_map="auto"` and can spread the model over multiple visible GPUs, but CPU offload is substantially slower. Model weights are downloaded from Hugging Face on first use.

`hybrid-vtg doctor` is read-only and checks the submodule, Python packages, FFmpeg, CUDA visibility, GPU name, and VRAM without loading either checkpoint.

## Datasets

Charades-STA accepts the original SemVID layout or a conventional `videos/` layout:

```text
Charades/
├── sta_annotation/charades_sta_test.txt
└── rgb_videos_30fps_480/*.mp4
```

ActivityNet-Grounding accepts:

```text
ActivityNet/
├── captions/val_2.json
└── rgb_videos_15fps_short256/*.mp4
```

Run either benchmark locally:

```bash
export CHARADES_STA_ROOT=/datasets/Charades
bash hybrid_vtg/scripts/run_charades_local.sh --limit 10 --fail-fast

export ACTIVITYNET_ROOT=/datasets/ActivityNet
bash hybrid_vtg/scripts/run_activitynet_local.sh --limit 10 --fail-fast
```

Remove `--limit 10` for a complete run. Results are append-only JSONL and resume by sample ID. Each run also writes an immutable manifest containing every method setting and the exact SemVID Git revision, plus a metrics JSON with mIoU and R@1 at IoU 0.3/0.5/0.7.

The main ablations use the same runner:

```bash
# Dense Qwen3-VL on the whole sampled video
bash hybrid_vtg/scripts/run_charades_local.sh \
  --no-temporal-prune --no-spatial-prune --no-refine \
  --output outputs/ablations/dense.jsonl

# SemVID alone
bash hybrid_vtg/scripts/run_charades_local.sh \
  --no-temporal-prune --no-refine \
  --output outputs/ablations/semvid.jsonl

# Temporal routing alone, with dense local Qwen prefill
bash hybrid_vtg/scripts/run_charades_local.sh \
  --no-spatial-prune --no-refine \
  --output outputs/ablations/temporal-only.jsonl
```

Use a different output path for every configuration because the adjacent manifest is immutable by design.

For another benchmark or a single video, use `--benchmark jsonl` with rows of this form:

```json
{"id":"sample-1","video_path":"/data/video.mp4","duration":123.4,"query":"the person opens the refrigerator","targets":[[42.1,47.3]],"group":"custom"}
```

`targets` may be omitted for inference-only runs.

## Benchmark plan

| Benchmark | Current adapter | Role in the paper |
|---|---:|---|
| Charades-STA | native | short indoor actions; direct SemVID comparison |
| ActivityNet-Grounding/Captions | native | diverse, longer untrimmed activities |
| QVHighlights | generic JSONL | longer videos and highlight-style query evidence |
| Ego4D-NLQ | generic JSONL | egocentric long-video search |
| TACoS | generic JSONL | fine-grained cooking actions |
| MAD | generic JSONL | very long movie grounding stress test |
| DiDeMo, YouCook2, TVR | generic JSONL | secondary transfer evaluation |

The primary report should include dense Qwen3-VL, SemVID alone, temporal routing alone, and the full hybrid at matched frame/token budgets. Report candidate recall before grounding, expert-encoded duration, retained visual tokens, wall time, peak VRAM, mIoU, and recall. Do not infer end-to-end speedup from token ratio alone.

## Important behavior

- Coarse features are query-independent and cached by video fingerprint, model, FPS, and frame cap.
- Window ranking uses fixed mean/peak cosine similarity, temporal NMS, a union-duration budget, halos, and a uniform low-confidence fallback.
- SemVID processes every retained connected component and produces true sparse Qwen prefill tokens. Proposed spans are selected with the same frozen coarse evidence; refinement is applied only to the selected span.
- Qwen is explicitly prompted for original-video timestamps. `--timestamp-mode relative` is available for model/checkpoint variants that emit clip-relative time.
- Refinement never reads annotations and never leaves the routed component.
- The default coarse cap is 2,048 frames. On extremely long videos this lowers the effective scan FPS instead of exceeding the fixed memory budget.

The main failure mode is cascaded recall: if temporal routing removes the target, SemVID cannot recover it. Candidate recall and retained-duration fraction must therefore be reported before final grounding accuracy.
