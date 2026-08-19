# Hybrid-VTG current progress report

**Date:** 2026-08-08<br>
**Research contract:** absolutely training-free<br>
**Recommended next direction:** [Budget-Conserving Semantic Corridor Routing](proposal_budget_conserving_semantic_corridor.md)

## Executive conclusion

The project has progressed from a coarse-to-fine idea into a runnable training-free VTG system based on frozen Qwen3-VL-4B-Thinking and the official SemVID spatial-pruning implementation. The system now has dataset adapters, temporal routing, spatial token pruning, timestamp normalization, boundary refinement, safe fallback, and detailed efficiency telemetry.

The present temporal method is not yet a successful contribution. On TACoS, weak SigLIP2 retrieval removed relevant evidence, independent component processing multiplied the vision and generation cost, and the final reranker sometimes discarded correct Qwen predictions. The safe fallback prevents catastrophic routing failures, but it fell back on every tested TACoS sample. Its observed behavior is therefore full-video SemVID plus coarse-search overhead.

This is not evidence that temporal sparsification is inherently unsuitable for VTG. The earlier 320-query OMTG study showed large gains from content-aware windows and frozen Qwen embedding retrieval. Together, the positive OMTG result and negative TACoS result identify a narrower research problem:

> Temporal sparsification must preserve a continuous evidence corridor, use a semantically strong retrieval representation, conserve one query-level compute budget, and decline to prune when its evidence is unstable.

The full sample-level diagnosis remains in [temporal_sparsification_failure_note.md](temporal_sparsification_failure_note.md).

## Frozen research contract

The final method must use no:

- supervised fine-tuning or reinforcement learning;
- optimizer, gradient update, adapter, or learned selector;
- task-specific grounding head;
- label-fitted calibration or benchmark-specific test threshold;
- ground-truth access outside evaluation.

All pretrained vision, embedding, and language models remain frozen in evaluation mode. Training-free does not mean unconstrained heuristic tuning: constants must be declared before final test evaluation and shared across benchmarks.

## Original research motivation

Long videos contain two structurally different forms of redundancy:

1. **Temporal redundancy:** most time intervals do not contain the queried event.
2. **Spatial redundancy:** even within a relevant interval, most patches are static, repetitive, background, or unrelated to the query.

The intended hierarchy is:

```text
whole-video temporal search
  -> continuous local evidence corridor
  -> spatial/token sparsification
  -> frozen temporal grounding
  -> local boundary refinement
```

Temporal selection is supposed to reduce frames and pixels reaching the expensive vision encoder. SemVID is supposed to reduce visual tokens reaching the language-model prefill and grounding stages. These are complementary only when both operate under one conserved query-level budget.

## What has been implemented

### Model and code migration

- TimeLens2 is retained as a legacy and OMTG comparison, not the primary Hybrid-VTG implementation.
- The primary grounder is the official pinned SemVID implementation with frozen `Qwen/Qwen3-VL-4B-Thinking`.
- The default spatial retention ratio is 12.5%.
- The current coarse encoder is frozen `google/siglip2-base-patch16-224`.

### Temporal processing

- low-FPS whole-video feature indexing;
- 8, 16, 32, and 64-second temporal proposals;
- post-halo marginal-coverage selection;
- adaptive context expansion;
- connected routed components with original-video timestamps;
- full-video fail-open routing when coarse confidence is weak.

### Grounding and refinement

- original-video absolute timestamp prompting and parsing;
- routed-component presence prompting;
- frozen Qwen interval generation;
- proposal tightness and boundary-contrast scoring;
- query-gated joint start/end refinement;
- deterministic output, resumption, and run manifests.

### Evaluation and efficiency support

- adapters and local launchers for TACoS, Charades-STA, and ActivityNet Grounding;
- OMTG/TimeLens2 controlled inference experiments;
- corrected router metrics based on target coverage, full containment, and endpoint availability;
- decoded frame, decoded pixel, vision time, dense/sparse prefill, component latency, fallback, and rejection telemetry;
- experimental microbatching, preprocessing overlap, and conditional refinement support.

## TACoS evidence

The following results are from the same ten-sample diagnostic subset, so they diagnose the implementation but are not benchmark-level claims.

| Configuration | R@1@0.3 | R@1@0.5 | R@1@0.7 | mIoU | Boundary MAE | Time/sample |
|---|---:|---:|---:|---:|---:|---:|
| Old temporal router + SemVID | 0.00 | 0.00 | 0.00 | 0.027 | 123.55 s | 30.71 s |
| Full-video SemVID upper bound | 0.90 | 0.60 | 0.50 | 0.595 | 2.58 s | 11.41 s |
| Safe full-video fallback | 0.70 | 0.50 | 0.50 | 0.556 | 6.42 s | 12.34 s |

The failed temporal route retained 43.3% of video duration but achieved only:

- 30% target full containment;
- 30% availability of both endpoints;
- 53.6% mean target coverage.

The safe version achieved full containment by falling back on every sample:

```text
TemporalFallbackRate = 1.0
mean retained duration fraction = 1.0
mean routed component count = 1.0
```

It is therefore not evidence that the current temporal stage improves accuracy or efficiency.

## Failure decomposition

### 1. The coarse semantic representation was too weak

At 0.5 FPS, global SigLIP2 frame/window similarity was nearly flat on repetitive TACoS cooking videos. Queries such as *take out*, *wash*, *cut*, and *place* differ through brief state transitions while sharing the same kitchen, person, and objects. The confidence fallback consequently produced approximately uniform components rather than meaningful query-dependent retrieval.

### 2. Target containment failed before grounding

Only 30% of targets had both endpoints available. Once the correct interval is removed, neither SemVID nor Qwen can recover it. Stage one must optimize safe evidence availability, not component/target IoU or aggressive duration compression.

### 3. Component processing multiplied the budget

Every routed component received a fresh pixel allowance. Eight short clips therefore cost substantially more than one continuous full-video call:

| Measurement | Eight routed clips | One full-video clip |
|---|---:|---:|
| Decoded frames | 268 | 498 |
| Decoded pixels | 276.8M | 32.1M |
| Vision-encoder time | 11.72 s | 1.96 s |
| Dense prefill tokens | 78,991 | 18,235 |
| SemVID prefill tokens | 11,447 | 4,509 |

Temporal routing decoded fewer frames but about 8.6 times more pixels and invoked Qwen as many as eight times. This invalidated the intended efficiency argument.

### 4. Irrelevant components were forced to produce intervals

The original prompt required timestamps for every clip. Qwen therefore generated plausible intervals even when the event was absent. A nullable component prompt is necessary for clipped inputs, but changing the full-video prompt caused a separate regression and must not be conflated with routing quality.

### 5. The final selector reused weak retrieval evidence

Timestamp parsing and component-to-global mapping were not the primary problem. For one query, Qwen predicted `[27, 32]` for target `[26.53, 33.54]`, but the final retrieval-based reranker selected `[218.81, 222.81]`. The router may determine where to spend compute; it must not override a grounded interval afterward using the same weak evidence that caused routing uncertainty.

### 6. Fragmentation removed the evidence chain

Independent components lose before/after state, action order, and continuous object trajectories. This is especially damaging for fine-grained state changes. More router coverage is not sufficient if grounding calls are fragmented and each receives too few local frames.

## Positive OMTG evidence

The TimeLens2 study evaluated complete 320-query runs using deterministic content-aware windows, frozen Qwen3-VL-Embedding-2B routing, and fixed 64/128-frame policies. The complete report is [results/comparison.md](../results/comparison.md).

For base Qwen3-VL-4B at 64 frames:

| Policy | C-Acc | EtF1 | tIoU |
|---|---:|---:|---:|
| Uniform one-shot | 1.25 | 1.04 | 28.05 |
| Embedding-window-local | 22.50 | 13.61 | 37.22 |

The embedding policy had 85.13% router recall. The paired video-cluster bootstrap favored embedding routing by 21.25 C-Acc, 12.57 EtF1, and 9.17 tIoU points. The 64-frame implementation had a small 148-frame aggregate overflow across 320 queries, which must not be repeated in the new method.

For TimeLens2-4B:

| Policy | C-Acc | EtF1 | tIoU |
|---|---:|---:|---:|
| Embedding-window-local, 64 frames | 36.25 | 18.48 | 40.94 |
| Embedding-window-local, 128 frames | 33.12 | 21.14 | 46.58 |

The later residual-search experiment increased router recall to 98.08% at 128 frames but reduced quality and increased calls and runtime. Its negative result, documented in [results/residual_search_report.md](../results/residual_search_report.md), reinforces the main TACoS lesson: exhaustive coverage and repeated local calls do not replace dense continuous grounding evidence.

## Current scientific position

The strongest current baseline is full-video SemVID. The present SigLIP temporal route should be treated as a failed ablation, not the proposed final method.

The project now has two defensible research paths:

1. [Budget-Conserving Semantic Corridor Routing](proposal_budget_conserving_semantic_corridor.md): repair pre-encoder temporal sparsification using stronger clip semantics, one continuous corridor, one global budget, and stability-gated fallback.
2. [Boundary Evidence Chain SemVID](proposal_boundary_evidence_chain_semvid.md): improve equal-token spatial allocation by explicitly preserving before/start/interior/end/after evidence without hard temporal deletion.

The two proposals must first be evaluated independently. Combining unvalidated temporal and spatial changes would make failures uninterpretable.

## Immediate experimental requirements

Before either proposal is claimed:

1. restore and freeze the full-video localization prompt baseline;
2. report dense Qwen, SemVID-only, temporal-only, and hybrid results under matched frames and pixels;
3. enforce a query-level budget rather than a per-component budget;
4. stratify all results by route versus fallback;
5. report cold indexing and cached-query inference separately;
6. use video-clustered paired bootstrap intervals;
7. evaluate complete benchmark splits after the diagnostic stage.

## Related project notes

- [Current strict-64 coarse-to-fine implementation](current_coarse_to_fine_implementation.md)
- [Current Mage and SemVID pruning implementation](current_pruning_implementation.md)
- [OMTG full-split Mage 0.8 + SemVID 0.25 model report](omtg_mage08_semvid025_model_report.md)
- [Temporal failure diagnosis](temporal_sparsification_failure_note.md)
- [Safe temporal routing implementation](next_implementation_safe_temporal_routing.md)
- [Original implementation plan](implementation_plan.md)
- [Codec-assisted SemVID idea](first_spatial_improvement_codec_assisted_semvid.md)
- [Boundary-evidence corridor](future_boundary_evidence_corridor.md)
- [Video saliency research note](video_saliency_for_training_free_vtg.md)
