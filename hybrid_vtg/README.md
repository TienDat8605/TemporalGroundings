# Hybrid-VTG: training-free temporal routing + SemVID

This package is the primary implementation of the proposed hierarchical VTG method. It replaces TimeLens2 as the research backbone and composes two frozen systems:

1. a cheap SigLIP2 whole-video scan finds high-recall temporal components;
2. the official SemVID Qwen3-VL implementation performs real object/motion/context token pruning and grounding inside every retained component;
3. SigLIP2 resamples a small neighborhood at high FPS and adjusts the predicted boundaries from semantic change and visual continuity.

No training code, optimizer, loss, adapter, checkpoint update, or benchmark-label access exists in the inference path. All models run in evaluation and inference mode.

## What is new relative to SemVID

SemVID sparsifies tokens after frames have reached the expert vision encoder. This package adds query-guided temporal routing before that expensive encoding. The research claim is therefore hierarchical conditional computation:

```text
whole video --cheap frozen scan--> retained temporal components
retained component --SemVID expert encoding + token pruning--> coarse interval
boundary neighborhoods --dense cheap scan--> refined global timestamps
```

SemVID remains pinned at `432a76928817cdfba7d04c460ac475482cd7c3a4`. Hybrid-VTG carries one reproducible patch that fixes compact-prefill attention masks, left padding, per-sample RoPE deltas, and telemetry for verified Qwen microbatching. The local run scripts apply it idempotently. TimeLens2 remains in the repository only as a legacy comparison baseline.

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

On a headless server, the preparation script downloads the public test-only mirror (1,334 videos, approximately 6.2 GiB compressed), resumes interrupted transfers, verifies checksums, checks ZIP paths before extraction, and validates every annotated video:

```bash
bash hybrid_vtg/scripts/prepare_charades_sta.sh \
  --accept-license \
  --data-root /root/datasets/Charades

source /root/datasets/Charades/activate_charades.sh
```

Review the [official Charades terms](https://prior.allenai.org/projects/charades) and the [public test mirror](https://huggingface.co/datasets/jwnt4/charades-sta-test) before passing `--accept-license`. The script removes the downloaded ZIP after successful validation to recover disk space; add `--keep-archive` to retain it. Set `CHARADES_VIDEOS_URL` or `CHARADES_ANNOTATION_URL` when using an authorized mirror.

ActivityNet-Grounding accepts:

```text
ActivityNet/
├── captions/val_2.json
└── rgb_videos_15fps_short256/*.mp4
```

Prepare the complete `val_2` split directly on the rented GPU server. The downloader
resumes its checksummed 39.1 GiB source archive, extracts only the 4,885 evaluation
videos, records missing IDs, and removes the archive after successful extraction:

```bash
bash hybrid_vtg/scripts/prepare_activitynet_grounding.sh \
  --accept-license \
  --data-root /root/datasets/ActivityNet

source /root/datasets/ActivityNet/activate_activitynet.sh
```

Review the [official ActivityNet terms](https://activity-net.org/download.html)
and the [public ActivityNet Captions mirror](https://huggingface.co/datasets/friedrichor/ActivityNet_Captions)
before passing `--accept-license`. At least 60 GB of temporary free space is
recommended. Add `--keep-archive` only if the 39.1 GiB source archive is needed
after preparation.

Run either benchmark on the GPU server:

```bash
export CHARADES_STA_ROOT=/datasets/Charades
bash hybrid_vtg/scripts/run_charades_local.sh --limit 10 --fail-fast

source /root/datasets/ActivityNet/activate_activitynet.sh
bash hybrid_vtg/scripts/run_activitynet_local.sh --limit 10 --fail-fast
```

Remove `--limit 10` for a complete run. Results are append-only JSONL and resume by sample ID. Each run also writes an immutable manifest containing every method setting and the exact SemVID Git revision, plus a metrics JSON with mIoU and R@1 at IoU 0.3/0.5/0.7.

## Optimized local inference

The default `safe` profile keeps batch-one execution, serial preprocessing, and fixed 8 FPS refinement. After validating equivalence on the rented GPU, enable the complete training-free optimization path with a fresh output name:

```bash
bash hybrid_vtg/scripts/run_charades_local.sh \
  --optimization-profile optimized \
  --output outputs/hybrid-vtg/charades-sta-optimized.jsonl
```

The optimized profile uses Qwen batch size two, pairs similar-duration components within a 16-sample look-ahead, prepares two microbatches in one CPU worker, and selects no/4/8 FPS boundary refinement from fixed endpoint-contrast percentiles. It does not read annotations when making any inference decision. Override individual stages for ablations:

```bash
# Microbatching only
bash hybrid_vtg/scripts/run_charades_local.sh \
  --qwen-batch-size 2 \
  --output outputs/ablations/qwen-batch-2.jsonl

# Microbatching plus CPU prefetch
bash hybrid_vtg/scripts/run_charades_local.sh \
  --qwen-batch-size 2 --preprocess-workers 1 --prefetch-depth 2 \
  --output outputs/ablations/qwen-batch-2-prefetch.jsonl

# Complete path with explicit settings
bash hybrid_vtg/scripts/run_charades_local.sh \
  --qwen-batch-size 2 --preprocess-workers 1 --prefetch-depth 2 --adaptive-refine \
  --output outputs/ablations/qwen-batch-2-prefetch-adaptive.jsonl
```

Use a different output path for every stage because optimization settings are part of the immutable run manifest. A batch-two CUDA OOM is retried as batch one and exposed as `qwen_oom_fallbacks`; any such fallback invalidates a strict speed benchmark.

For the 32–64 sample batching gate, add `--capture-validation-logits` to both batch-one and batch-two runs, then compare them:

```bash
hybrid-vtg validate-optimization \
  --baseline outputs/validation/batch-1.jsonl \
  --candidate outputs/validation/batch-2.jsonl \
  --mode equivalence --minimum-samples 32 --minimum-speedup 0.15
```

The equivalence gate requires identical parsed intervals, identical first-token argmax values, top-eight logit differences no greater than 0.05, no OOM fallback, and the requested speedup. Use `--mode refinement --minimum-samples 256 --minimum-speedup 0.08` for the adaptive-refinement gate; it enforces the predeclared 0.005 mIoU and one-percentage-point recall budgets without changing thresholds.

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

The primary report should include dense Qwen3-VL, SemVID alone, temporal routing alone, and the full hybrid at matched frame/token budgets. Report routed target coverage and endpoint availability before grounding, expert-encoded duration, decoded frames/pixels, retained visual tokens, vision/prefill latency, wall time, peak VRAM, mIoU, and recall. Do not infer end-to-end speedup from token ratio alone.

## Important behavior

- Coarse features are query-independent and cached by video fingerprint, model, FPS, and frame cap.
- Window ranking uses mean/peak cosine similarity, asymmetric uncertainty-aware halos, post-halo marginal coverage, a merged-component cap, a union-duration budget, and a uniform low-confidence fallback.
- SemVID processes every retained connected component and produces true sparse Qwen prefill tokens. Proposed spans are reranked by boundary contrast and interval evidence concentration rather than the retrieval score.
- Qwen is explicitly prompted for original-video timestamps. `--timestamp-mode relative` is available for model/checkpoint variants that emit clip-relative time.
- Refinement jointly selects valid endpoint pairs using query-gated visual change, inside/outside evidence contrast, and a duration prior. It never reads annotations or leaves the routed component.
- The default coarse cap is 2,048 frames. On extremely long videos this lowers the effective scan FPS instead of exceeding the fixed memory budget.
- Efficiency telemetry reports decoded frames/pixels, vision-encoder time, sparse/dense prefill lengths, per-component latency, end-to-end time, and peak VRAM.

The main failure mode is cascaded recall: if temporal routing removes the target, SemVID cannot recover it. Target coverage, endpoint availability, full containment, and retained-duration fraction must therefore be reported before final grounding accuracy.
