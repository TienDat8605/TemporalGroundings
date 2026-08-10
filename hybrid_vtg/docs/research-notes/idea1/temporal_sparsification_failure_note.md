# Temporal sparsification failure diagnosis

**Date:** 2026-08-06  
**Status:** Handoff note for the next session

## Main conclusion

Temporal sparsification did not fail as a general idea. The present Hybrid-VTG implementation failed because it combined weak frame-level retrieval, disconnected fallback windows, repeated expert-model calls, per-component pixel budgets, forced timestamp hallucination, and an unreliable final reranker.

Timestamp parsing or timestamp merging was not the primary failure. Qwen produced correct original-video timestamps inside several correct components, but the final selector discarded them. For example:

```text
Target:          [26.53, 33.54]
Qwen prediction: [27.00, 32.00]
Final output:    [218.81, 222.81]
```

## Evidence from TACoS

| Configuration | R@1 IoU 0.3 | R@1 IoU 0.5 | mIoU | Boundary MAE | Seconds/sample |
|---|---:|---:|---:|---:|---:|
| Old temporal router + SemVID | 0.00 | 0.00 | 0.027 | 123.55 s | 30.71 s |
| Full-video SemVID upper bound | 0.90 | 0.60 | 0.595 | 2.58 s | 11.41 s |
| Safe full-video fallback | 0.70 | 0.50 | 0.556 | 6.42 s | 12.34 s |

The old temporal route retained 43.3% of the video but achieved only:

- 30% target full containment;
- 30% availability of both endpoints;
- 53.6% average target coverage.

The safe implementation obtained 100% containment by falling back on every sample. Therefore, its current TACoS behavior is full-video SemVID plus coarse-search overhead, not active temporal sparsification.

## Failure decomposition

### 1. Weak TACoS retrieval signal

Frozen SigLIP2 scans at 0.5 FPS and compares global frame/window appearance with the query. TACoS contains continuous, repetitive cooking activity with the same person, kitchen, tools, and objects. Queries often differ through short state transitions such as taking, washing, cutting, or placing an object. Their SigLIP scores were too flat to support reliable routing.

The old confidence fallback consequently selected approximately uniform windows around 22, 50, 78, 105, 133, 162, 190, and 218 seconds. This was not meaningful semantic retrieval.

### 2. Per-component budget multiplication

Every routed component received a fresh Qwen video pixel budget. Splitting the input therefore increased total processing despite decoding fewer frames:

| Measurement | Eight routed clips | One full-video clip |
|---|---:|---:|
| Decoded frames | 268 | 498 |
| Decoded pixels | 276.8M | 32.1M |
| Vision-encoder time | 11.72 s | 1.96 s |
| Dense prefill tokens | 78,991 | 18,235 |
| SemVID prefill tokens | 11,447 | 4,509 |

The routed version decoded about 8.6 times more pixels and used multiple Qwen generations. Temporal selection must conserve a global query-level frame, pixel, and token budget rather than resetting that budget for every component.

### 3. Too many expert calls

The current router grounded as many as eight components independently and selected a result afterward. Each component repeated preprocessing, vision encoding, SemVID selection, Qwen prefill, and generation. A useful temporal hierarchy should cheaply rank many candidates but ground at most one or two continuous candidates.

### 4. Forced hallucination and weak final selection

The old prompt required an interval inside every component, even when the event was absent. Qwen therefore invented plausible timestamps for irrelevant clips. Boundary-only reranking then used the same weak SigLIP evidence and sometimes discarded correct Qwen intervals.

The new presence prompt permits abstention, but full-video TACoS samples all produced `presence_score=1.0`; this run did not validate negative-component rejection or confidence calibration. The presence prompt also reduced the full-video result relative to the simpler localization prompt.

### 5. Lost continuity

Independent short clips remove action order, preceding object state, and before/after context. Full-video SemVID retains enough continuous evidence to distinguish repeated fine-grained cooking steps while still pruning spatial tokens.

## Why the TimeLens2 experiment may succeed

The earlier TimeLens2 experiment used PySceneDetect candidate clips, a frozen Qwen-2B embedding reranker, and a single 64-frame budget shared by ranking and grounding. Compared with the current implementation, it likely benefits from:

1. content-aligned rather than arbitrary overlapping candidates;
2. stronger clip/action semantics from the Qwen embedding model;
3. one global frame budget rather than a fresh budget per component;
4. grounding only the best evidence instead of every candidate;
5. fewer model invocations;
6. possible better alignment between scene structure and OMTG videos.

PySceneDetect alone may not solve TACoS because continuous fixed-camera cooking videos can contain few shot changes. Its segmentation should be combined with a maximum clip length or temporal subwindows.

## Revised temporal direction

```text
Global 64–96 frame/pixel budget
        -> cheap whole-video candidate construction
        -> frozen Qwen embedding clip reranking
        -> retain at most 1–2 continuous evidence corridors
        -> allocate the remaining global budget locally
        -> one Qwen grounding call when possible
        -> optional SemVID pruning inside the same global budget
```

Required properties:

- one query-level frame, pixel, and token budget;
- original timestamps attached to every candidate;
- high endpoint-containment recall before grounding;
- explicit event absence for clipped candidates;
- no coarse SigLIP boundary score as the dominant final selector;
- temporal continuity around the selected event;
- separate reporting for fallback and genuinely pruned samples.

## First experiments for the next session

### 1. Add controlled prompt modes

Implement:

```text
localize: timestamp-only legacy prompt
presence: presence verification plus timestamps
auto:     localize for full-video fallback, presence for clipped candidates
```

Use the localization prompt to restore the full-video SemVID upper bound. Prompt testing fixes the fallback regression; it does not fix temporal retrieval.

### 2. Run the four core sparsification ablations

| Temporal pruning | SemVID | Meaning |
|---|---|---|
| No | No | Dense Qwen baseline |
| No | Yes | SemVID-only baseline |
| Yes | No | Temporal-only method |
| Yes | Yes | Complete hybrid |

A temporal-only result is not meaningful when `TemporalFallbackRate=1.0`; in that case it is dense Qwen plus routing overhead.

### 3. Cross the TimeLens2 and Hybrid-VTG routers

Test:

1. PySceneDetect + frozen Qwen-2B embedding router with SemVID/Qwen3-VL on TACoS;
2. current SigLIP multi-scale router with TimeLens2 on OMTG;
3. original TimeLens2 router and grounder on OMTG;
4. current Hybrid-VTG pipeline on TACoS.

This separates the effects of segmentation, retrieval model, global budget, grounding model, and dataset structure.

### 4. Add matched-budget diagnostics

Before comparing accuracy, record for each query:

- candidate target coverage and full containment;
- number of expert components and Qwen generations;
- globally allocated frames and pixels;
- dense and sparse prefill tokens;
- oracle best component prediction versus selected prediction;
- fallback rate and non-fallback accuracy;
- wall time and peak VRAM.

## Research interpretation

Full-video SemVID is currently the strong TACoS baseline and safety path. The temporal contribution becomes valid only when it retains a smaller continuous region with high endpoint recall and reduces total query-level compute. The most promising lesson from TimeLens2 is not scene detection alone; it is stronger clip-level semantic ranking under one conserved global budget.
