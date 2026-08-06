# Implementation Plan — Training-Free Hierarchical SemVID-VTG

## Frozen research contract

The method is absolutely training-free: no SFT, GRPO, optimizer, gradients, learned router, fitted boundary head, parameter update, or label-dependent threshold calibration. Public pretrained checkpoints remain frozen under evaluation and inference mode. Ground truth is visible only to evaluation code.

> First determine where the event may occur, then preserve only the visual evidence needed to recognize it and refine its boundaries.

## Migrated architecture

TimeLens2 is no longer the primary implementation because its public wrapper accepts selected frames but cannot consume externally pruned visual tokens. It remains a legacy baseline. The primary code is `hybrid_vtg/`, based on the official, pinned SemVID implementation.

```text
Raw long video
  -> frozen SigLIP2 scan at low FPS
  -> multi-scale query/window retrieval
  -> confidence gate
       -> reliable: budgeted temporal selection + adaptive halos
       -> uncertain: one continuous full-video fallback component
  -> high-detail decode of retained continuous component(s)
  -> official SemVID query/object/motion/context token sparsification
  -> frozen Qwen3-VL event-presence verification and grounding in global video seconds
  -> presence-first component selection
  -> high-FPS frozen-feature boundary refinement
  -> final [start, end]
```

This separates three costs:

- temporal routing prevents discarded frames from reaching the expensive expert vision encoder;
- SemVID reduces the retained visual sequence before language-model prefill;
- local refinement spends dense sampling only around two predicted boundaries.

SemVID’s spatial policy already implements query-conditioned frame allocation, diverse object-token selection through MMR, query-aware motion relay, and context anchors. The project must use that implementation rather than presenting the same signals as a new spatial-pruning contribution. The distinct contribution is the pre-encoder temporal hierarchy, global timestamp protocol, boundary refinement, and joint compute/accuracy analysis.

## Fixed default configuration

| Role | Default |
|---|---|
| Cheap router | `google/siglip2-base-patch16-224`, frozen |
| Coarse sampling | 0.5 FPS, at most 2,048 frames |
| Windows | 8, 16, 32, 64 seconds; half-window stride |
| Window score | 0.5 pooled cosine + 0.5 peak frame cosine |
| Temporal budget | 120 seconds of halo-expanded union |
| Halo | 2 seconds per side |
| Sparse grounder | official SemVID + `Qwen/Qwen3-VL-4B-Thinking`, frozen |
| Visual-token retention | 12.5% |
| SemVID token mix | 60% object; remaining motion/context protection |
| Boundary refinement | 8 FPS within 2 seconds of predicted endpoints |
| Generation | deterministic, temperature zero |

These settings are versioned in the run manifest. Any development sweep must be declared before final test evaluation and reported completely.

## Implementation ownership

| Module | Responsibility |
|---|---|
| `video.py` | timestamp sampling, probing, and exact frame decoding |
| `coarse_encoder.py` | frozen SigLIP image/text features |
| `index.py` | reusable query-independent coarse cache |
| `temporal.py` | multi-scale scoring, NMS, budgets, halos, fallback |
| `semvid_bridge.py` | official sparse-Qwen loading, clipped inference, telemetry |
| `timestamps.py` | parsing and absolute/relative coordinate normalization |
| `refinement.py` | semantic-change and continuity boundary adjustment |
| `benchmarks.py` | Charades-STA, ActivityNet, and canonical JSONL adapters |
| `pipeline.py` | end-to-end orchestration without model-policy duplication |
| `cli.py` | local GPU runs, resume, manifests, and evaluation |

The SemVID submodule is never edited. Its exact commit is recorded for each run.

## Evaluation plan

Primary datasets are Charades-STA and ActivityNet-Grounding, matching SemVID. QVHighlights and Ego4D-NLQ test longer search; TACoS tests fine-grained actions; MAD is the long-video stress test. DiDeMo, YouCook2, and TVR are secondary transfer benchmarks. Non-native datasets enter through a canonical JSONL schema until native adapters are added.

Required comparisons:

1. dense frozen Qwen3-VL;
2. SemVID alone on the whole sampled video;
3. temporal routing with dense local Qwen3-VL;
4. temporal routing + SemVID;
5. full hybrid with boundary refinement;
6. uniform temporal selection under matched duration/frame budgets;
7. motion-only and query-only routing diagnostics.

Required measurements:

- temporal candidate recall before grounding;
- expert-encoded seconds and frames;
- original/retained SemVID tokens and prefill sequence length;
- end-to-end wall time and component-level latency;
- peak allocated/reserved GPU memory;
- mIoU and R@1 at IoU 0.3, 0.5, and 0.7;
- boundary error and refinement acceptance/gain;
- accuracy versus total retained visual-token volume.

Token retention is not an end-to-end speed claim. Router, decoding, vision, token selection, prefill, generation, and refinement time must be reported separately.

## Safety and ablation rules

- The coarse cache cannot include query or annotation data.
- Low score concentration triggers one continuous full-video SemVID pass. It does not trigger disconnected uniform windows that can lose short events and multiply generation overhead.
- Routed-component prompts permit an explicit `present=false` response; the frozen model is never forced to invent a timestamp for an irrelevant component.
- Positive proposals are ranked first by frozen-Qwen presence confidence, then boundary quality, then coarse retrieval score.
- Halos are charged on their deduplicated union.
- SemVID receives the raw query separately because its pruning model uses query tokens.
- Predicted timestamps are clamped to the routed component and original video duration.
- Refinement cannot move a boundary outside the retained component.
- Failed timestamp parsing is recorded; it is never replaced by ground truth.
- All ties use deterministic ordering.

## Milestones

1. **Migration:** pin SemVID, create standalone package, preserve TimeLens2 baseline.
2. **Core inference:** coarse index, router, SemVID clipped grounding, global timestamp parsing.
3. **Refinement and telemetry:** endpoint resampling, token counts, retained-duration accounting.
4. **Benchmark readiness:** Charades/ActivityNet adapters, generic JSONL, resumable local launchers.
5. **Scientific validation:** unit tests, smoke test on one clip, dense/SemVID/hybrid equivalence and ablations.

The first GPU smoke test must use one short clip and verify that SemVID reports fewer retained than original video tokens, the returned interval is in original-video seconds, and no parameter has `requires_grad=True`. Full benchmark runs start only after this gate passes.
