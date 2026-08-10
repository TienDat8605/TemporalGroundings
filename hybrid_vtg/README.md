# TPSA: timeline-preserving spatial allocation

This package benchmarks a frozen Qwen3-VL grounder with one continuous full video and one generation per query. TPSA retains at least one real visual token from every post-encoder temporal tubelet and distributes the remaining exact token budget using query relevance, feature-native transitions, and directional boundary evidence.

## Implementation status

TPSA currently prunes **after** the frozen Qwen vision encoder. All sampled frames and spatial patches still pass through every vision-transformer block.

| Stage | TPSA | HMVE Phase A |
| --- | --- | --- |
| Observation selection | One full-resolution pass | Low-resolution full-timeline scout, then one full-frame corridor pass |
| Within the vision encoder | No pruning | No pruning; spatial crops and mid-encoder pruning are deferred |
| Before LLM prefill | Exact post-encoder selection | Cross-pass deduplication and exact evidence packing |
| LLM generations | One | One |

The retention ratio therefore measures post-encoder visual tokens delivered to the language model. It reduces multimodal prefill length but does not reduce video decoding or frozen vision-encoder computation. Pre-encoder and mid-encoder hierarchical pruning are roadmap ideas, not part of the results produced by this package.

Spatial policies:

- `dense`: all post-encoder visual tokens;
- `uniform`: equal tokens per frame;
- `semvid`: official SemVID selection baseline;
- `tpsa_query`: timeline coverage plus query evidence;
- `tpsa_motion`: query evidence plus feature-native state change;
- `tpsa_boundary`: complete TPSA allocator.

TPSA v3 constructs both auxiliary policies from the exact `tpsa_query`
selection. At most 10% of its non-prototype tail is unlocked. Motion uses
adjacent same-cell feature differences blended equally with positive query
relevance, cannot change per-frame counts, and falls back exactly on weak or
flat motion. `tpsa_boundary` splits the same pool evenly between motion and
directional boundary evidence; only boundary evidence may move tokens between
frames. Its prominence gate is median plus two MAD, with one-second evidence
windows, four-second NMS, at most four bands per direction, and two-sided
coverage. Boundary timing uses the post-encoder tubelet FPS, not decoded FPS.

SemVID remains the command-line default until the declared promotion gates pass. The removed temporal router, component reranker, presence verifier, and refinement path are not part of this implementation.

HMVE is a separate observation policy enabled with `--observation-policy
hmve`. Phase A performs exactly two frozen vision-encoder calls: a low-FPS,
low-resolution full-timeline scout and one normal-resolution call containing
all expanded query-relevant corridors. Scout projections are cached, at least
one global anchor from every scout tubelet survives, redundant coarse evidence
can be replaced by detailed evidence, and the accumulated pack is sorted by
absolute timestamp and compacted to the declared retention budget before one
LLM generation. HMVE currently supports Qwen batch size one.

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

# The single permitted TPSA v3 diagnostic: repaired boundary only, IDs 0--63
bash hybrid_vtg/scripts/run_omtg_local.sh \
  --limit 64 \
  --spatial-policy tpsa_boundary \
  --retention-ratio 0.125 \
  --output outputs/tpsa-v3-omtg-diagnostic/tpsa_boundary-0p125.jsonl \
  --fail-fast

# Compare with the archived query control; exit 0 means promote, exit 2 means HMVE fallback
hybrid-vtg validate-tpsa-v3 \
  --control results/tpsa-v2-omtg-diagnostic/tpsa_query-0p125.jsonl \
  --candidate outputs/tpsa-v3-omtg-diagnostic/tpsa_boundary-0p125.jsonl
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

# Three TPSA stages at 12.5%; completed baselines are skipped
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

# HMVE Phase A: two vision observations, one exact 12.5% evidence pack,
# one language-model generation
hybrid-vtg run \
  --benchmark omtg \
  --data "$OMTG_ROOT" \
  --split test \
  --observation-policy hmve \
  --spatial-policy tpsa_query \
  --qwen-batch-size 1 \
  --retention-ratio 0.125 \
  --output outputs/omtg-hmve-query-0p125.jsonl
```

Use a distinct output path for each configuration because the adjacent manifest is immutable. Manifests use schema 7 and record the observation policy, spatial policy, allocator constants, project revision, and SemVID revision.

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

Every sample records target and actual retained tokens, effective retention, per-frame allocation, token role counts, selected boundary bands, query-only overlap, attempted and actual replacements, motion-gated frames, rejected boundary evidence, median/MAD/peak evidence summaries, quota returned to query, allocator-stage latency, original and compact prefill lengths, decoded frames/pixels, vision time, generation time, end-to-end time, and peak VRAM. HMVE additionally reports corridors, per-pass decoded pixels, encoder calls and estimated FLOPs, created and retained tokens, scout-cache reuse, cross-pass replacement, controller latency, and the single LLM call. Deprecated upstream `semvid_*` statistics are retained beside neutral spatial fields for existing analysis scripts.

Single-span datasets report mIoU, boundary MAE, and R@1 at IoU 0.3/0.5/0.7. OMTG uses its native multi-interval targets and set-valued metrics. Re-score an existing JSONL with:

```bash
hybrid-vtg evaluate --input outputs/omtg.jsonl
```

The evaluator accepts numeric-pair and object-form JSON, recovers complete
intervals from truncated generations, applies the shared duplicate/gap merge,
and reports `parse_status_counts`. This also repairs older TPSA JSONLs whose raw
object-form responses were retained but whose `intervals` field was empty.

TPSA is post-encoder: all policies decode identical frames and run the same frozen vision encoder. Any speed claim must therefore use measured latency rather than infer savings from retained-token ratio.
