# First spatial improvement: codec-assisted, motion-compensated SemVID

> **Recommendation:** implement this before pre-encoder patch sparsification or a full boundary-evidence corridor. Keep the official SemVID token budget and its object/context protections unchanged, and improve only how temporal change and motion tokens are scored.

## Why this should come first

Mage-VL demonstrates an important principle: local temporal predictability is a useful measure of visual information. Its codec front-end retains dense anchor frames and selects predicted-frame patches using motion vectors and residual energy before the vision transformer. This saves vision-encoder compute, but Mage-ViT is trained from scratch to consume irregular sparse patch sequences with preserved 3D coordinates.

Directly applying the same pre-ViT sparsity to frozen Qwen3-VL is risky. Qwen's vision encoder was trained on continuous image/video grids, and aggressive irregular patch removal could create a representation and positional-distribution shift. It would also make it difficult to determine whether a grounding change came from a better selection signal or from an incompatible encoder input.

The first implementation should therefore import Mage-VL's **information signal**, not its sparse encoder architecture:

```text
temporally routed continuous clip
  -> normal dense Qwen vision encoding
  -> codec-assisted SemVID scoring at the existing pruning point
  -> unchanged SemVID token count and role safeguards
  -> frozen Qwen grounding
```

This remains completely training-free and preserves the input format that the frozen Qwen vision encoder expects.

## Scope of the first implementation

Name the experimental variant **Codec-Assisted Motion-Compensated SemVID (CAM-SemVID)**. It should be an optional mode beside the unchanged official SemVID baseline.

Do not change these elements in the first experiment:

- frozen SigLIP temporal router;
- continuous routed components and original timestamps;
- Qwen vision input, patch embedding, or vision-transformer layers;
- total SemVID retention ratio;
- object-token MMR;
- per-frame context floor;
- frozen Qwen weights and deterministic generation;
- current grounding and refinement logic.

Change only the motion/change evidence used in SemVID's frame-budget and motion-token calculations.

## Step-by-step design

### 1. Extract codec information without decoding extra model frames

For each temporally retained component, extract from the source video where possible:

- I-frame and P-frame locations;
- block motion vectors;
- prediction/reference-frame indices;
- a local residual-energy or reconstruction-novelty estimate.

Map these signals onto the spatial grid used by SemVID. Multiple codec blocks falling inside one visual patch should be robustly pooled. Store the result as timestamped metadata so repeated queries over the same video can reuse it.

The implementation must not require a particular codec. Use the following fallbacks:

1. exported H.264/HEVC motion vectors plus residual estimate;
2. optical flow plus motion-compensated pixel residual;
3. existing SemVID feature difference if neither is available.

Record which path was used in run telemetry.

### 2. Remove dominant camera motion

Raw codec motion is not semantic motion. Panning, zooming, stabilization, and cuts can create strong motion everywhere.

Estimate dominant global motion using a robust median motion field or a fitted affine/homography model. Define local motion as the deviation from that global field:

\[
m^{\mathrm{local}}_{t,p}
=
\left\|
\operatorname{MV}_{t,p}-\operatorname{MV}^{\mathrm{global}}_t(p)
\right\|.
\]

Scene cuts should be handled separately. A cut may justify a context anchor, but it must not automatically receive a high query-motion score.

### 3. Compute motion-compensated feature change

SemVID currently measures patch change mainly at the same grid location across neighboring frames. For a moving object, the corresponding content is usually at another location.

Use the codec or optical-flow correspondence to compare a patch with its predicted source:

\[
d^{\mathrm{mc}}_{t,p}
=
\left\|
\hat V_{t,p}
-
\hat V_{t-1,\,\operatorname{warp}(p,\operatorname{MV}_{t,p})}
\right\|_2.
\]

This distinguishes ordinary translation from an actual appearance or state change. Preserve the existing same-location feature difference as a fallback and as an ablation.

### 4. Form a codec-novelty score

Normalize all terms within each shot or continuous component, not globally across the benchmark:

\[
n_{t,p}
=
\lambda_r r_{t,p}
+
\lambda_m m^{\mathrm{local}}_{t,p}
+
\lambda_d d^{\mathrm{mc}}_{t,p},
\]

where \(r_{t,p}\) is residual/reconstruction novelty. Start with declared fixed weights or equal normalized weights. Do not fit them to VTG labels.

### 5. Modify SemVID motion-token scoring conservatively

Combine codec novelty with SemVID's existing query relevance:

\[
s^{\mathrm{CAM}}_{t,p}
=
\beta_q s^{\mathrm{query}}_{t,p}
+
(1-\beta_q)n_{t,p}.
\]

Query relevance remains essential because a codec assigns bits to all unpredictable changes, including irrelevant people, background screens, water, and camera artifacts.

Only candidates assigned to SemVID's motion-token role should use this new score. Object tokens should continue to use query similarity and MMR, and context tokens should keep their existing safeguards.

### 6. Improve frame-level budget allocation without changing total tokens

Replace or augment the coarse inter-frame-change term with pooled, camera-compensated codec novelty:

\[
w_t
=
\alpha s^{\mathrm{query}}_t
+
(1-\alpha)\operatorname{Pool}_p(n_{t,p}).
\]

Use a robust top-quantile or trimmed-mean pool so a small meaningful transition can influence the frame budget without one corrupted patch dominating it. Retain SemVID's minimum per-frame context quota and exact global token budget.

### 7. Add anchor/update allocation only after the scoring ablation

Mage-VL's dense-anchor/sparse-update pattern is valuable, but introducing it simultaneously with new motion scores would confound the experiment.

After CAM scoring is validated, test a conservative **post-encoder** anchor/update policy:

- ordinary frames retain the normal SemVID context floor;
- component starts and genuine scene transitions receive modestly larger context budgets;
- future boundary-corridor start/end bands receive protected anchor budgets;
- no frame becomes token-empty;
- total token retention stays matched to the SemVID baseline.

Do not use codec I-frames blindly as semantic anchors. Their locations are selected for compression efficiency, not query relevance or event boundaries.

## Required implementation structure

Keep codec extraction independent of SemVID so it can be cached and tested separately:

```text
codec metadata extractor
  -> timestamped motion/residual maps
  -> grid alignment and global-motion compensation
  -> CAM score provider
  -> optional SemVID scoring adapter
```

The official SemVID submodule must remain pinned and unmodified. Implement the behavior in our adapter/fork so the baseline can always be reproduced exactly.

Telemetry should record:

- codec and extraction backend;
- number of I/P frames and valid motion blocks;
- extraction and alignment latency;
- cache hit/miss;
- average raw and camera-compensated motion;
- residual/novelty distribution;
- per-frame object, motion, and context-token counts;
- vision time, pruning time, prefill time, and total time.

## Evaluation and ablation order

Run all comparisons with the same temporal components, decoded frames, SemVID token ratio, prompt, Qwen checkpoint, and generation settings.

1. Official SemVID.
2. SemVID + residual novelty only.
3. SemVID + raw codec motion and residual.
4. SemVID + camera-compensated codec novelty.
5. SemVID + motion-compensated feature difference.
6. Full CAM-SemVID.
7. CAM-SemVID + conservative anchor/update allocation.
8. Later: CAM-SemVID + boundary-evidence corridor.

Report:

- mIoU and R@1 at IoU 0.3/0.5/0.7;
- start, end, and mean boundary error;
- results separated by camera-motion, shot-cut, subtle-action, and static-object cases;
- retained-token role counts and exact token ratio;
- end-to-end latency and component-level overhead;
- prediction changes relative to official SemVID.

The first success criterion is not speed. At this stage, dense vision encoding remains unchanged. Success means better boundary accuracy or robustness at the same SemVID token budget with small, measured codec-extraction overhead.

## Explicitly deferred

Do not include these in the first implementation:

- removing arbitrary Qwen patches before the vision-transformer trunk;
- packing irregular patches into Mage-style canvases;
- changing Qwen's positional encoding;
- learned codec/token gates;
- fitting fusion weights on VTG annotations;
- aggressive reduction below the established SemVID token ratio;
- non-contiguous video montages presented as continuous time.

A pre-ViT sparse-Qwen experiment is only justified after CAM-SemVID and boundary corridors are understood, and only if batch-one equivalence, timestamp behavior, positional handling, and matched-compute accuracy are validated. It should remain an optional systems experiment rather than the main training-free claim.

## Relationship to the boundary-evidence corridor

CAM-SemVID supplies a better local change signal; the boundary corridor supplies the temporal role of that change. Their eventual combination should be:

\[
\text{query evidence}
+
\text{camera-compensated local novelty}
+
\text{boundary role and uncertainty}.
\]

The recommended order is therefore:

```text
official SemVID baseline
  -> CAM motion/change scoring at matched tokens
  -> boundary-evidence corridor
  -> boundary-conditioned anchor/update budgets
  -> optional pre-ViT sparsity feasibility study
```

## Sources motivating the design

- [Mage-VL: An Efficient Codec-Native Streaming Multimodal Foundation Model](https://arxiv.org/html/2607.24904v1)
- [SemVID: Keeping the Evidence Chain](https://arxiv.org/html/2603.05663v4)

