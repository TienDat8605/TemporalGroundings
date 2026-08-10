# Proposal 1: Budget-Conserving Semantic Corridor Routing

**Short name:** BC-SCR<br>
**Status:** recommended next research implementation<br>
**Contract:** frozen models and deterministic inference only

## Research question

Can a frozen semantic router reduce pre-encoder video compute without degrading temporal grounding when it is constrained to select one continuous evidence corridor, conserve one query-level budget, and fall back when its decision is unstable?

## Central hypothesis

Temporal pruning is useful only when it changes the allocation of evidence rather than multiplying independent model executions. A semantically reliable continuous corridor should let the frozen grounder spend a fixed frame/pixel budget at higher local temporal density. When retrieval is ambiguous, a matched-budget full-video pass should be safer than disconnected candidate calls.

The method is:

```text
query-independent content-aware video index
  -> frozen query/window embedding similarity
  -> stability and margin gate
       -> confident: one continuous semantic corridor
       -> uncertain: full-video uniform fallback
  -> one globally budgeted Qwen call
  -> optional official SemVID pruning inside that call
  -> final timestamp interval
```

## Why this is a solution to the observed failure

| Current failure | BC-SCR response |
|---|---|
| Flat SigLIP scores on procedural actions | Frozen Qwen3-VL-Embedding-2B clip semantics |
| Arbitrary multiscale windows | Content-aware 20–60-second windows |
| 30% endpoint availability | Stability-gated hard route and safe fallback |
| Eight independent grounder calls | Exactly one primary grounding call |
| Per-component pixel reset | One query-level frame/pixel/token ledger |
| Lost temporal continuity | One connected corridor with context |
| Correct Qwen answer discarded | No retrieval-based final reranking |
| Fallback prompt regression | Explicit prompt mode based on input type |

## Method specification

### 1. Content-aware candidate construction

Construct query-independent windows once per video:

1. detect content transitions with deterministic PySceneDetect `ContentDetector(threshold=27)` and `frame_skip=4`;
2. pack adjacent scenes into windows between 20 and 60 seconds;
3. overlap neighboring packed windows by two seconds;
4. when no useful boundaries are detected, use deterministic 45-second windows with four-second overlap;
5. preserve source-video start/end timestamps for every window.

Scene boundaries are structural hints, not event predictions. The maximum-duration rule remains active because TACoS may contain few cuts.

For a video with too many windows for direct indexing, coalesce adjacent windows into at most eight scene-aligned macro-regions, score those regions, and expose the children of the two highest-scoring macro-regions. Score at most eight children. Both hierarchy levels are recorded as cold-index work.

### 2. Frozen semantic index

Use frozen Qwen3-VL-Embedding-2B as the router instead of SigLIP2.

For every candidate window:

- sample four temporally uniform frames;
- form one embedding from the first and third frames;
- form a second embedding from the second and fourth frames;
- store both normalized embeddings and the mean embedding;
- cache video-side embeddings without the query or annotations.

At query time, compute one frozen text embedding and cosine similarities against the cached video embeddings. The two interleaved representations provide a query-time stability test without extra video decoding.

### 3. Stability-gated routing

Let the mean-embedding scores be `s`. Define:

\[
m = \frac{s_{(1)} - s_{(2)}}{P_{90}(s)-P_{10}(s)+10^{-6}}.
\]

Activate hard temporal routing only when:

1. the same window ranks first under both interleaved indices; and
2. `m >= 0.5`.

These values are fixed before benchmark evaluation and are not fitted to target timestamps.

If either condition fails, use the full-video fallback. Record each failed condition separately so fallback is scientifically interpretable.

### 4. Continuous corridor construction

When routing is accepted:

1. start with the highest-scoring content window;
2. include adjacent scene fragments until retained duration is at least 20 seconds;
3. add four seconds on both sides for pre/post evidence;
4. clamp to the original video;
5. cap total duration at 64 seconds by retaining context nearest the selected window;
6. decode the result as one continuous clip in original chronological order.

The main single-span method selects only one corridor. Multi-span OMTG transfer uses a separately named set-valued mode and cannot be mixed into the single-span headline result.

### 5. Query-level compute conservation

Evaluate two grounder budgets: 64 and 128 frames.

- Full-video fallback receives the complete frame budget uniformly across the video.
- A routed corridor receives the same complete frame budget uniformly within the corridor.
- All frames use the same declared maximum pixels per frame.
- `total_pixel_budget = frame_budget × maximum_pixels_per_frame`.
- There is exactly one primary Qwen generation per query.
- SemVID, when enabled, retains 12.5% of visual tokens from that call.
- No stage or component may reset the frame, pixel, or token budget.

Cached retrieval and online grounding have separate ledgers. Report:

- **cold cost:** scene detection, router decoding, router vision encoding, query embedding, and grounding;
- **cached-query cost:** query embedding, ranking, grounding, and optional refinement;
- **amortized cost:** cold index cost divided by the observed number of queries sharing a video, plus cached-query cost.

### 6. Prompt and timestamp protocol

Expose three prompt modes:

- `localize`: one interval in original-video seconds;
- `nullable_localize`: one interval or explicit absence;
- `auto`: `localize` for full-video fallback and `nullable_localize` for a routed corridor.

Predictions are clamped to the source video and, for routed inference, to the corridor. The corridor offset is applied exactly once. Timestamp parsing failures remain failures and are never replaced with target information.

### 7. Final decision

The single Qwen interval is the final pre-refinement prediction. Retrieval similarity must not rerank or replace it. If boundary refinement is enabled, it may change the interval only when its joint inside-versus-outside objective improves by at least 0.01.

## Interfaces and required telemetry

### Configuration

```text
router_backend: qwen_embedding
temporal_policy: semantic_corridor
global_frame_budget: 64 | 128
maximum_grounder_calls: 1
routing_margin: 0.5
corridor_min_seconds: 20
corridor_max_seconds: 64
corridor_context_seconds: 4
prompt_mode: localize | nullable_localize | auto
semvid_retention_ratio: 0.125
```

### Per-query output

Record:

- candidate windows and both phase scores;
- top-one/top-two scores, robust margin, and phase agreement;
- route or fallback decision and reason;
- retained corridor and duration fraction;
- target coverage/full containment in evaluation output only;
- router, grounder, and refinement frames/pixels;
- dense and retained visual tokens;
- Qwen call count and per-stage latency;
- cold, cached, and amortized cost summaries;
- peak VRAM and budget violations.

## Evaluation design

### Datasets

Primary single-span evaluation:

- TACoS: repetitive procedural actions and fine boundaries;
- Charades-STA: short indoor actions;
- ActivityNet Grounding: longer and more diverse events.

Secondary transfer:

- OMTG in an explicitly set-valued mode;
- long-video stress testing only after the primary method is stable.

### Required baselines

1. dense full-video frozen Qwen;
2. full-video official SemVID;
3. current SigLIP temporal router with dense Qwen;
4. current SigLIP router with SemVID;
5. TimeLens-style top-K independent calls;
6. BC-SCR without SemVID;
7. BC-SCR with official SemVID;
8. target-oracle corridor as a diagnostic upper bound.

Every accuracy comparison uses matched grounder frames and maximum pixels. The two-FPS full-video setting is an additional reference, not a matched-budget baseline.

### Essential ablations

- content-aware versus fixed-duration windows;
- SigLIP2 versus Qwen3-VL-Embedding-2B;
- one corridor versus top-K independent calls;
- stability gate versus unconditional top-one routing;
- global versus per-component budget accounting;
- full-video versus corridor at 64 and 128 frames;
- dense versus 12.5% SemVID retention;
- cold versus cached-query execution;
- localization, nullable, and automatic prompts.

### Metrics

- R@1 at IoU 0.3, 0.5, and 0.7;
- mIoU and start/end MAE;
- target coverage, full containment, and endpoint availability;
- route rate, fallback rate, and accuracy stratified by decision;
- decoded frames/pixels and vision-encoder time;
- dense and retained prefill tokens;
- Qwen calls, end-to-end time, and peak VRAM;
- accuracy versus total compute Pareto curves.

Use paired percentile-bootstrap confidence intervals clustered by video with 10,000 resamples and a declared fixed seed.

## Acceptance and falsification criteria

The proposal is supported only if:

- non-fallback corridor full-containment recall is at least 85%;
- there are zero frame and pixel budget violations;
- the single-span path averages exactly one primary grounding call;
- BC-SCR + SemVID is within two absolute points of matched-budget full-video SemVID on R@1@0.5 and mIoU;
- it reduces amortized end-to-end time by at least 25%, or establishes a statistically supported superior accuracy/compute Pareto point;
- R@1@0.3 falls by no more than one absolute point.

The proposal is falsified as a general hard-routing solution if stable routes cannot achieve 85% containment, if most queries still fall back, or if corridor density fails to compensate for lost global context under matched pixels. That negative result should redirect the project to [Boundary Evidence Chain SemVID](proposal_boundary_evidence_chain_semvid.md), which avoids hard temporal deletion.

## Scientific contribution and positioning

BC-SCR is not presented as the first training-free router. Training-free and hierarchical selection already exist. The narrower contribution is:

> A failure-aware, query-level budget policy for composing pre-encoder temporal routing with post-encoder spatial pruning, while preserving one continuous evidence corridor and one primary frozen-grounder call.

This differs from [SemVID](https://arxiv.org/abs/2603.05663), which primarily reduces post-vision visual tokens, and from [TimeLens2](https://arxiv.org/abs/2607.17423), whose principal contributions concern temporal-grounding data and interval-set optimization. The OMTG implementation provides evidence and reusable engineering, but BC-SCR targets safe single-span VTG and explicitly studies routing/fallback under matched end-to-end budgets.

## Implementation order

1. restore the full-video `localize` upper bound;
2. add the query-level resource ledger and enforce one call;
3. port content-aware windows and frozen Qwen embedding indexing;
4. add phase stability and robust-margin fallback;
5. implement continuous corridor allocation;
6. run router-only oracle/containment diagnostics;
7. run temporal-only matched-budget experiments;
8. add unchanged official SemVID and run the complete hybrid;
9. execute full datasets only after the acceptance gates pass on fixed diagnostic subsets.

## Related notes

- [Current progress report](current_progress_report.md)
- [Temporal failure diagnosis](temporal_sparsification_failure_note.md)
- [Safe temporal routing](next_implementation_safe_temporal_routing.md)
- [OMTG comparison](../results/comparison.md)
- [Residual-search negative result](../results/residual_search_report.md)
