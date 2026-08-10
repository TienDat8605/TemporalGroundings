# Future work: boundary-evidence corridors for Hybrid-VTG

> **Status: deferred.** Do not implement or present this as the current method until the existing Hybrid-VTG pipeline has been benchmarked fairly against SemVID, dense Qwen3-VL, and relevant temporal-grounding SOTAs. The first priority is to establish the current method's accuracy, routing recall, boundary error, and end-to-end compute trade-offs.

> **Implementation order:** before building the full corridor, first validate the conservative codec-assisted SemVID change described in [first_spatial_improvement_codec_assisted_semvid.md](first_spatial_improvement_codec_assisted_semvid.md). It improves local change evidence at SemVID's existing post-encoder pruning point without exposing frozen Qwen's vision encoder to irregular sparse patch inputs.

## Motivation

The present pipeline has a useful division of labour:

1. frozen SigLIP2 retrieval removes temporal regions before the expensive expert vision encoder;
2. SemVID allocates post-encoder visual tokens to context, objects, and motion within each retained component;
3. a frozen local refinement stage can adjust the final endpoints.

SemVID's spatial pruning and Hybrid-VTG's temporal routing are complementary. However, neither one explicitly reserves visual evidence that proves *when an event starts and ends*. A fixed halo gives surrounding context, but it does not distinguish relevant onset/offset evidence from generic context or unrelated shot changes.

## Proposed direction

Replace fixed-window expansion with a **boundary-evidence corridor**. Every retained continuous component should contain five explicit temporal roles:

```text
pre-boundary contrast | start anchor | event interior | end anchor | post-boundary contrast
```

The goal is not merely to retrieve frames that depict the event. It is to retain enough before/after evidence to discriminate the event interval from visually similar activity immediately outside it.

## Training-free design sketch

Use only frozen models and cached coarse features.

1. Compute coarse query-presence scores over the whole video.
2. Estimate directional boundary evidence:
   - start evidence: a sustained rise in query relevance across a local split;
   - end evidence: a sustained fall in query relevance across a local split;
   - gate visual-change cues by those directional semantic signals, so a shot cut alone cannot define a boundary;
   - when available, use camera-compensated codec residual and motion-compensated feature change rather than raw same-location frame difference.
3. For each retrieval candidate, create a connected corridor with start/end anchors and contrast bands. Allocate larger left/right context when endpoint uncertainty is high, rather than using a fixed symmetric halo.
4. Preserve the corridor as a continuous clip with original-video timestamps. Do not create a non-contiguous montage unless timestamp/temporal-position handling is solved explicitly.
5. Extend or fork SemVID only at its frame/token allocation interface:
   - reserve a minimum context budget in start/end anchor bands;
   - increase query-object allocation near likely boundaries;
   - prioritize motion tokens only where camera-compensated local novelty agrees with directional query-evidence change;
   - treat codec I-frames only as compression anchors, not automatic semantic or boundary anchors.
   SemVID's existing context/object/motion token selection should remain the spatial-pruning baseline.
6. Score generated intervals using a frozen interval objective, not coarse retrieval score alone:

   ```text
   interior relevance + start contrast + end contrast
   - outside relevance - excessive-length penalty
   ```

7. Refine start/end jointly over local candidates. Accept a change only if the complete interval objective improves and the interval remains inside its corridor.

## Why this differs from the current method

The current method performs hard temporal routing, SemVID pruning, then independent local endpoint adjustments. The proposal introduces a shared boundary representation before expert inference and carries it through token allocation and final ranking. It targets a narrow, testable claim: boundary-aware evidence retention improves temporal localization at matched compute.

## Relationship to codec-assisted spatial evidence

The two ideas solve different parts of the boundary problem:

- codec-assisted motion compensation estimates **what locally changed** while discounting predictable translation and dominant camera motion;
- the boundary corridor estimates **whether that change supports an event start or end** for the current query.

Codec novelty must not define a boundary by itself. It remains query-agnostic and can respond strongly to cuts, background activity, or unrelated objects. The eventual boundary token score should combine query evidence, codec/local novelty, and the frame's pre/start/interior/end/post role.

The safe development order is:

```text
official SemVID
  -> codec-assisted motion scoring at the same token budget
  -> semantic boundary corridor
  -> boundary-conditioned anchor/update token allocation
  -> optional pre-ViT sparse-input feasibility study
```

Pre-ViT patch sparsification is not part of the corridor's initial implementation. Mage-style irregular inputs are only a future systems experiment because Mage-ViT is specifically trained for sparse codec-token sequences, whereas frozen Qwen expects its normal dense visual front-end.

## Required evidence before implementation

Complete and document current-method comparisons first:

- Dense frozen Qwen3-VL, SemVID alone, temporal routing alone, and current full Hybrid-VTG.
- Relevant published or openly runnable SOTAs under compatible data, prompt, and frame/token budgets.
- Router target coverage, full-containment recall, and endpoint-availability recall. Component/target IoU alone is not an adequate router-recall measure.
- Boundary error before and after current refinement, separated by shot-cut and non-shot-cut cases.
- Decoder frames/pixels, vision-encoder time, token-selection time, prefill length/time, generation time, peak memory, and end-to-end wall time.

Only proceed if these results show a material boundary-specific failure mode that the present pipeline cannot fix by tuning the existing training-free routing, SemVID settings, or refinement protocol.

## Future ablations

If implemented, evaluate at matched total compute:

1. current router with fixed halos;
2. adaptive corridors without SemVID boundary priors;
3. SemVID boundary-prior allocation without adaptive corridors;
4. full boundary-evidence corridor;
5. semantic-only, visual-change-only, and combined boundary signals;
6. independent versus joint endpoint refinement.

Report accuracy, boundary MAE, endpoint availability, retained duration, pre-encoder frames/pixels, post-encoder tokens, and full runtime. Do not infer end-to-end speedup from retained token ratio alone.
