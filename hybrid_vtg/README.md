# TPSA: timeline-preserving spatial allocation

This package benchmarks a frozen Qwen3-VL grounder with one continuous full video and one generation per query. TPSA retains at least one real visual token from every decoded frame and distributes the remaining exact token budget using query relevance, feature-native transitions, and directional boundary evidence.

Spatial policies:

- `dense`: all post-encoder visual tokens;
- `uniform`: equal tokens per frame;
- `semvid`: official SemVID selection baseline;
- `tpsa_query`: timeline coverage plus query evidence;
- `tpsa_motion`: query evidence plus feature-native state change;
- `tpsa_boundary`: complete TPSA allocator.

SemVID remains the command-line default until the declared promotion gates pass. The removed temporal router, component reranker, presence verifier, and refinement path are not part of this implementation.

## Setup

```bash
git submodule update --init --recursive
conda create -n hybrid-vtg python=3.11 -y
conda activate hybrid-vtg
pip install -r hybrid_vtg/requirements.txt
pip install -e 'hybrid_vtg[test]'
hybrid-vtg doctor
```

The default checkpoint is `Qwen/Qwen3-VL-4B-Thinking`. Inference requires CUDA; SDPA is the default attention backend.

## 1. OMTG Bench first

[OMTG Bench](https://huggingface.co/datasets/insomnia7/omtg_bench) is the primary bring-up benchmark. Its 320 questions require set-valued output: the model is asked for every disjoint occurrence, and evaluation reports cardinality accuracy, temporal IoU, Hungarian-matched temporal precision/recall/F1, and effective temporal F1. Review its CC BY-NC 4.0 terms and the source-video licenses before downloading.

```bash
bash hybrid_vtg/scripts/prepare_omtg.sh --data-root /datasets/OMTGBench
export OMTG_ROOT=/datasets/OMTGBench

# Smoke test
bash hybrid_vtg/scripts/run_omtg_local.sh --limit 10 --fail-fast

# Dense plus every equal-token policy at 6.25%, 12.5%, and 25%
bash hybrid_vtg/scripts/run_spatial_matrix.sh \
  omtg outputs/tpsa-matrix/omtg --fail-fast
```

Remove `--limit` for the complete fixed test set. Each result is append-only and resumes by sample ID.

## 2. Compressed TACoS second

TACoS uses [VideoMind's complete compressed release](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos): 127 videos at 3 fps, 480p, without audio, plus `train.jsonl`, `val.jsonl`, and `test.jsonl`. The preparation script is pinned to a specific dataset revision, validates annotation hashes and row counts, checks archive paths before extraction, verifies all referenced video IDs, and rejects unexpected frame rates, dimensions, or audio streams.

```bash
bash hybrid_vtg/scripts/prepare_tacos.sh \
  --data-root /datasets/TACoS-compressed
export TACOS_ROOT=/datasets/TACoS-compressed

# Smoke test
bash hybrid_vtg/scripts/run_tacos_local.sh --limit 10 --fail-fast

# Equal-token matrix
bash hybrid_vtg/scripts/run_spatial_matrix.sh \
  tacos outputs/tpsa-matrix/tacos --fail-fast
```

The downloaded video artifact is `tacos/videos_3fps_480_noaudio.tar.gz` (about 1.49 GB), not the 30.2 GB original-video archive. Add `--keep-archive` only if the local compressed tarball is still needed after extraction.

## Direct runs

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data "$OMTG_ROOT" \
  --split test \
  --spatial-policy tpsa_boundary \
  --retention-ratio 0.125 \
  --output outputs/omtg-tpsa-boundary-0p125.jsonl
```

Use a distinct output path for each configuration because the adjacent manifest is immutable. Manifests use schema 4 and record the spatial policy, allocator constants, project revision, and SemVID revision.

For batch-two and CPU prefetch after validating on the target GPU:

```bash
bash hybrid_vtg/scripts/run_omtg_local.sh \
  --optimization-profile optimized \
  --output outputs/omtg-optimized.jsonl
```

Capture logits in matched batch-one and batch-two runs, then apply the strict equivalence gate:

```bash
hybrid-vtg validate-optimization \
  --baseline outputs/validation/batch-1.jsonl \
  --candidate outputs/validation/batch-2.jsonl \
  --minimum-samples 32 \
  --minimum-speedup 0.15
```

The gate requires identical parsed interval sets, matching first-token argmax, top-eight logit differences no greater than 0.05, no CUDA-OOM fallback, and the requested wall-clock speedup.

## Outputs and evaluation

Every sample records target and actual retained tokens, effective retention, per-frame allocation, token role counts, selected boundary bands, allocator-stage latency, original and compact prefill lengths, decoded frames/pixels, vision time, generation time, end-to-end time, and peak VRAM. Deprecated upstream `semvid_*` statistics are retained beside neutral spatial fields for existing analysis scripts.

Single-span datasets report mIoU, boundary MAE, and R@1 at IoU 0.3/0.5/0.7. OMTG uses its native multi-interval targets and set-valued metrics. Re-score an existing JSONL with:

```bash
hybrid-vtg evaluate --input outputs/omtg.jsonl
```

TPSA is post-encoder: all policies decode identical frames and run the same frozen vision encoder. Any speed claim must therefore use measured latency rather than infer savings from retained-token ratio.
