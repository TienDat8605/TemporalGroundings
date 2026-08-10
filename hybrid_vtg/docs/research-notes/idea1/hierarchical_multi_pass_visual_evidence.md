# Hierarchical Multi-Pass Visual Evidence Accumulation

- **Status:** Three-pass temporal controller implemented in `hybrid_vtg`;
  spatial zoom, in-encoder pruning, and adaptive stopping remain proposals
- **Working name:** HMVE (Hierarchical Multi-Pass Visual Encoder)
- **Primary benchmark:** OMTG Bench at 12.5% final visual-token retention
- **Model contract:** frozen Qwen3-VL vision encoder and frozen LLM;
  training-free

## 1. Core idea

Instead of encoding the video once at full cost and pruning visual tokens only
afterward, run the frozen vision encoder several times as a coarse-to-fine
observer:

1. scan the complete timeline cheaply;
2. identify query-relevant temporal corridors;
3. revisit those corridors at higher temporal detail;
4. refine likely event boundaries with the finest pass;
5. accumulate the useful visual evidence from every pass; and
6. give the final evidence pack to the LLM in exactly one generation.

The implemented method prunes **temporally between passes**. Spatial pruning
between passes and pruning **inside the vision encoder** are later extensions.
It does not discard all coarse evidence after zooming in: a small set of global
timeline anchors is retained so the LLM can interpret the detailed evidence in
the context of the whole video.

```text
query + full video
        |
        v
Pass 0: cheap global scan -----------------------------------+
  0.5 FPS, full-frame, full timeline                        |
        | relevance / novelty / uncertainty                  |
        v                                                    |
Pass 1: candidate refinement --------------------------+     |
  1 FPS, temporal corridors, full-frame                |     |
        | content and boundary evidence                 |     |
        v                                               |     |
Pass 2: fine boundary observation ----------------+     |     |
  2 FPS near relevance rises/falls, full-frame     |     |     |
        |                                          |     |     |
        +----------------------+-------------------+-----+-----+
                               v
                    evidence accumulator
             deduplicate, replace, order, budget
                               |
                               v
                       one LLM generation
                               |
                               v
                     global interval prediction
```

## 2. Why this differs from the previous hierarchy

The earlier hierarchical test-time search runs the complete grounder on many
windows and merges several text predictions. HMVE instead makes several
**vision-only** observations and performs **one** language-model call.

| Property | Earlier window search | HMVE |
|---|---|---|
| Vision encoder calls | Multiple | Multiple |
| LLM generations | Multiple | One |
| Intermediate result | Predicted intervals | Visual evidence |
| Global context | Reconstructed by interval merge | Preserved by global anchors |
| Main risk | Inconsistent text predictions | Coarse pass misses relevant evidence |
| Main cost | Repeated encoder and LLM work | Repeated encoder work only |

This also differs from current TPSA. TPSA makes one dense vision-encoder pass
and allocates post-encoder tokens. HMVE tries to avoid encoding every location
at full temporal and spatial detail in the first place.

## 3. Three-pass observation policy

The values below are initial experimental defaults, not learned parameters.
Selection is deterministic and uses no benchmark labels for fitting.

### Pass 0: global scout

Purpose: maximize recall over the entire timeline at low cost.

- Decode the complete video at 0.5 FPS using the shared full-frame decoder.
- Run one complete frozen vision-encoder call over the batched scout frames.
- Preserve at least one real global anchor for every sampled temporal unit.
- Score temporal units using the backend's training-free query similarity.
- Produce several temporal corridors rather than one winning interval, because
  OMTG queries can have multiple disjoint occurrences.
- Keep one actual scout token per sampled temporal unit in the final pack as
  the uniform exploration reserve.

The scout is allowed to be imprecise. Its job is to avoid false negatives and
decide where the next observation should spend pixels and encoder FLOPs.

### Pass 1: corridor and object refinement

Purpose: recognize the event and localize the relevant actors or objects.

- Expand every candidate into a 16-second corridor.
- Decode the union of those corridors at 1 FPS.
- Encode all full-frame corridor observations in one chronological batch and
  one vision-encoder call.
- Recompute query relevance from the medium-detail evidence.
- Retain multiple disjoint corridors when their evidence remains competitive.

Spatial ROIs are intentionally deferred. Full-frame context avoids losing
relations such as “person enters the room” or “object moves from the table to
the sink.”

### Pass 2: boundary refinement

Purpose: resolve the start and end of each surviving occurrence.

- For every corridor, locate the largest query-relevance rise and fall in the
  Pass-1 temporal score sequence.
- Build a two-second halo on each side of those locations.
- Decode the union of all boundary bands at 2 FPS, the densest rate used by
  HMVE.
- Encode all full-frame boundary observations in one chronological batch and
  one vision-encoder call.

This pass supplies visual evidence, not an intermediate timestamp prediction.
The final LLM still decides the interval set.

## 4. Evidence accumulator

The current accumulator records per-pass frame counts and evidence blocks plus
one absolute timestamp per visual unit. The complete planned provenance is:

```text
(absolute_time, spatial_box, source_resolution, pass_id,
 encoder_depth, evidence_role, score, original_position_id)
```

Evidence roles include:

- `global_anchor`: coarse coverage of the complete timeline;
- `content`: query-relevant appearance or interaction;
- `motion`: local state change after camera-motion compensation;
- `start`: before-to-after relevance rise;
- `end`: before-to-after relevance fall; and
- `exploration`: evidence outside current candidates.

The accumulator must not simply concatenate all passes. It applies four rules:

1. **Preserve coverage.** Never remove every token representing a sampled
   global temporal unit.
2. **Replace redundant detail.** A fine token may replace overlapping coarse
   content evidence, but it may not remove the last global anchor for that
   time.
3. **Deduplicate.** Suppress tokens that overlap strongly in time, space, and
   feature similarity unless their evidence roles differ.
4. **Enforce one final budget.** The LLM receives exactly the declared number
   of visual tokens, independent of how many encoder passes were made.

The final evidence pack is sorted by absolute time. Qwen compact prefill keeps
explicit position IDs, while timestamp labels preserve the original absolute
seconds instead of treating selected corridors as adjacent video.

## 5. Training-free controller

The controller is a deterministic state machine, not a learned policy:

```text
state_0 = GlobalScout(video, query, scout_budget)
corridors_1, rois_1 = ProposeObservations(state_0)

state_1 = Encode(video, corridors_1, rois_1, medium_detail)
corridors_2, rois_2 = RefineObservations(state_0, state_1)

state_2 = Encode(video, corridors_2, rois_2, fine_detail)
evidence = Accumulate(state_0, state_1, state_2)

visual_tokens = SelectExactBudget(evidence, final_token_budget)
intervals = GenerateOnce(visual_tokens, query)
```

The implemented selection signal is the same label-free query similarity used
by TPSA. Appearance residuals, motion, uncertainty, and dedicated directional
boundary features remain planned extensions:

- maximum patch-to-query cosine similarity;
- feature-native appearance and local-motion residuals;
- directional start/end evidence; and
- uncertainty or flatness, used to increase exploration rather than to delete
  the timeline.

When query and intermediate vision features have different dimensions, use an
existing frozen Qwen projection or merger output. Do not train a new selector.

## 6. Spatial pruning inside the encoder

The safest first implementation prunes only between complete encoder calls.
The next version can prune at an intermediate vision block:

1. run early blocks on all patches in the current observation;
2. score spatial merge cells using query similarity, novelty, and coverage;
3. retain whole merge cells rather than arbitrary child patches;
4. run later blocks only on the retained cells; and
5. pass the surviving cells through the original frozen merger.

For Qwen3-VL, a selected spatial merge cell must keep all children required by
the configured merger (normally a 2-by-2 group). This preserves reshape and
merger invariants. Original rotary positions are gathered for the selected
children, per-frame cumulative sequence lengths are recomputed, and DeepStack
features use the identical selection.

The coverage guarantee at this stage applies to each Qwen temporal tubelet,
which may represent more than one decoded raw frame.

## 7. Budgets and honest compute accounting

HMVE has two separate budgets:

- **observation budget:** decoded pixels and total vision-encoder FLOPs across
  every pass;
- **reasoning budget:** final retained visual tokens and LLM prefill length.

A 12.5% final token budget does not by itself mean 12.5% end-to-end compute.
Every run must record:

- decoded frames and pixels per pass;
- temporal coverage and crop area per pass;
- vision blocks executed and estimated/measured vision FLOPs;
- tokens created, accumulated, replaced, and finally retained;
- cache hits for repeated frames or regions;
- vision, controller, prefill, generation, and end-to-end latency;
- number of encoder calls and the single LLM call; and
- peak VRAM.

Repeated observations should reuse decoded frames, patch embeddings, and early
block features when their sampling grid and crop are identical. Cache reuse is
an optimization and must be reported rather than assumed.

## 8. Failure safeguards

- **Scout miss:** reserve uniform exploration tokens and expand candidate
  corridors with margins.
- **Multi-occurrence collapse:** keep several disjoint candidates and apply
  temporal NMS, never a single hard argmax.
- **Wrong ROI:** include low-resolution full-frame context beside every crop.
- **Boundary tunnel vision:** retain before/during/after evidence around each
  proposed boundary.
- **Camera motion:** remove median displacement before ranking local motion.
- **Evidence explosion:** use deterministic caps and one exact final budget.
- **Timeline corruption:** store absolute seconds and original position IDs for
  every observation.
- **Non-determinism:** fix all tie-breaking by time, spatial index, and pass ID.
- **Dense regression:** a bypass mode must reproduce the ordinary single-pass
  dense model when pruning is disabled.

If the scout evidence is flat or uncertain, the controller falls back to broad
uniform coverage. It must never interpret uncertainty as permission to remove
most of the video.

## 9. Incremental implementation plan

### Phase A: three-pass temporal prototype — implemented

- Reuse the existing video decoder and frozen Qwen vision encoder.
- Implement Pass 0 at low FPS, Pass 1 on temporal corridors, and Pass 2 around
  relevance-rise and relevance-fall boundary evidence.
- Keep full-frame views; defer spatial crops and mid-encoder pruning.
- Accumulate projected encoder outputs with absolute timestamps.
- Pack an exact 12.5% token budget and call the LLM once.

This phase answers the central question with the smallest architectural change:
does three-pass accumulated coarse-to-fine visual evidence outperform one-pass
TPSA at the same final LLM token count?

### Phase B: spatial zoom

- Derive ROIs from patch relevance and transition evidence.
- Add crop plus context observations.
- Deduplicate coarse and fine evidence by provenance and similarity.

### Phase C: in-encoder pruning

- Add a selection hook at one audited intermediate vision block.
- Select complete spatial merge cells and recompute attention metadata.
- Cache early features reused by later observations.

### Phase D: adaptive stopping

- Stop when candidate corridors, boundary bands, and evidence ranks stabilize,
  subject to a strict maximum pass and compute budget.
- Keep the stopping rule fixed and training-free for the primary study.

## 10. Initial OMTG experiment

Reuse the existing completed dense, uniform, and SemVID results; do not rerun
those baselines. At 12.5% final retained visual tokens, compare the archived
results with:

1. `tpsa_query`;
2. `tpsa_motion`;
3. `tpsa_boundary`;
4. HMVE Phase A; and
5. HMVE with spatial zoom.

All methods use identical queries, source videos, timestamp prompt, frozen
weights, and one LLM generation for the primary comparison. Report both:

- accuracy at equal final LLM token count; and
- accuracy versus total measured inference cost, including all HMVE vision
  passes.

Primary OMTG metrics remain effective temporal F1, cardinality accuracy, and
temporal F1 at IoU 0.3/0.5/0.7. Boundary MAE and candidate-corridor recall are
diagnostics. Only after the OMTG design is stable should the same fixed policy
be tested on compressed 3-fps/480p TACoS.

## 11. Research hypothesis

At the same final LLM visual-token budget, a frozen multi-pass encoder can
provide better grounding evidence than a single dense encode followed by
post-encoder pruning because it spends high-resolution temporal and spatial
computation only where coarse evidence indicates it is useful.

The expected benefit is sharper boundaries and better recognition of brief or
small events. The claim is valid only if the gain remains favorable after all
extra vision passes, decoded pixels, and controller latency are counted.
