# Proposal 2: Boundary Evidence Chain SemVID

**Short name:** BEC-SemVID<br>
**Status:** independent spatial research direction<br>
**Contract:** equal-token, training-free inference with a frozen dense vision encoder

## Research question

Can an equal-token, training-free spatial allocator improve precise temporal grounding by explicitly preserving the ordered evidence needed to establish an event's start and end?

## Central hypothesis

Existing semantic pruning is effective at preserving frames and patches that depict an event, but temporal grounding needs more than event presence. It needs a causal visual sequence:

```text
before state -> onset transition -> event interior -> offset transition -> after state
```

High query similarity commonly peaks in the event interior. Query-only selection can therefore keep the action while removing the contrast needed to locate its boundaries. Generic motion is also insufficient because camera movement, shot cuts, and unrelated activity can dominate it.

BEC-SemVID reserves a complete **boundary evidence chain** under exactly the same retained-token budget as official SemVID. It preserves continuous full-video context and does not expose frozen Qwen's vision encoder to irregular pre-ViT sparse inputs.

## Why this is a solution rather than an exploratory score

The proposal changes the representation delivered to the frozen language model in a controlled way:

- start and end evidence are directional rather than symmetric visual change;
- motion is corrected for camera motion and gated by query evidence;
- before/after context receives a guaranteed budget;
- boundary candidates remain multiple and ordered rather than becoming an early hard interval;
- endpoints are optimized jointly using interval-level evidence;
- total retained tokens are identical to the SemVID baseline.

The core paper question is consequently causal and falsifiable: does explicit evidence-chain preservation improve boundary accuracy at equal token count?

## Method specification

### 1. Preserve the frozen dense visual front-end

The full sampled video is passed through the normal frozen Qwen vision encoder. BEC-SemVID operates at SemVID's existing post-encoder pruning point.

Do not:

- remove raw patches before the vision transformer;
- alter Qwen positional encodings;
- train a sparse encoder;
- change the model weights;
- change the retained-token count relative to the matched SemVID run.

This avoids the distribution shift that would occur if a frozen dense video encoder were given Mage-style irregular sparse patch grids.

### 2. Construct normalized evidence signals

For frame `t` and patch `p`, compute five signals:

1. `q(t,p)`: patch/query relevance from frozen normalized visual and text features;
2. `o(t,p)`: official SemVID object/context importance;
3. `n(t,p)`: local visual novelty after camera-motion compensation;
4. `b_start(t)`: directional query-evidence rise;
5. `b_end(t)`: directional query-evidence fall.

Convert each signal into a percentile rank within its shot or continuous video component. Rank normalization prevents one signal's raw scale from dominating and removes the need for label-fitted calibration.

### 3. Camera-compensated local novelty

Use the following deterministic backend order:

1. codec motion vectors and residual/reconstruction energy for supported H.264/HEVC video;
2. optical flow and motion-compensated pixel residual;
3. nearest-neighbor visual-feature correspondence;
4. existing same-location SemVID feature difference.

Estimate dominant camera motion with a robust affine model; if fitting is unstable, use the median motion field. Define local motion as deviation from the predicted global field. Scene cuts create context anchors but do not automatically become event boundaries.

For a patch correspondence `warp(p)`:

\[
d^{mc}_{t,p}=\left\|\hat V_{t,p}-\hat V_{t-1,warp(p)}\right\|_2.
\]

Fuse within-shot percentile ranks of residual energy, local motion, and motion-compensated feature change with an unweighted mean. Query relevance is applied later, so codec novelty cannot independently define importance.

The conservative codec implementation is elaborated in [first_spatial_improvement_codec_assisted_semvid.md](first_spatial_improvement_codec_assisted_semvid.md).

### 4. Directional boundary evidence

Let `Q(t)` be a robust top-quantile pool of patch/query relevance. With a one-second evidence window, compute:

\[
\Delta^+(t)=Q_{after}(t)-Q_{before}(t),
\qquad
\Delta^-(t)=Q_{before}(t)-Q_{after}(t).
\]

Then define:

\[
b_{start}(t)=rank(\max(0,\Delta^+(t)))\,rank(N(t)),
\]

\[
b_{end}(t)=rank(\max(0,\Delta^-(t)))\,rank(N(t)),
\]

where `N(t)` is pooled local novelty. Require the relevance change to persist for at least one second so isolated noisy frames do not become anchors.

Apply temporal non-maximum suppression with a four-second exclusion radius. Retain up to four start bands and four end bands. Every band spans one second before and after its peak. These bands are evidence candidates; they do not form the final interval before Qwen inference.

### 5. Exact equal-token allocation

Let `K` be the number of tokens retained by official SemVID at the selected ratio. BEC-SemVID must retain exactly `K` tokens.

At the default 12.5% ratio:

- **20% context pool:** periodic whole-video anchors plus pre/post context around candidate bands;
- **20% boundary pool:** 10% start-band tokens and 10% end-band tokens;
- **60% adaptive pool:** percentile-rank fusion of query relevance, official object importance, context importance, and camera-compensated novelty.

Round quotas deterministically and assign any remainder to the adaptive pool. If a role has fewer eligible tokens than its quota, transfer the unused quota to the adaptive pool.

Additional safeguards:

- retain at least one merged token group per sampled frame;
- preserve official SemVID object diversity/MMR behavior in the adaptive pool;
- preserve original chronological order and temporal positions;
- do not let a scene cut consume boundary quota without directional query evidence;
- merge redundant background tokens using the official SemVID mechanism rather than replacing them with arbitrary zeros.

### 6. Frozen grounding

Run one full-video Qwen grounding call with the legacy `localize` prompt and original-video timestamps. The input remains temporally continuous. BEC-SemVID changes only which post-vision visual evidence survives under the fixed token budget.

### 7. Joint endpoint refinement

When refinement is enabled:

1. resample two seconds around the predicted start and end at eight FPS;
2. enumerate valid start/end pairs with `start < end`;
3. score each pair using interior relevance, start rise, end fall, inside-versus-outside contrast, and a weak duration-deviation penalty;
4. gate visual continuity by query contrast so unrelated cuts cannot attract endpoints;
5. accept the best pair only when it improves the complete interval objective by at least 0.01;
6. otherwise retain Qwen's original interval.

The exact boundary-corridor logic and risks are discussed in [future_boundary_evidence_corridor.md](future_boundary_evidence_corridor.md).

## Interfaces and required telemetry

### Configuration

```text
spatial_policy: boundary_evidence_chain
motion_backend: auto | codec | optical_flow | feature_correspondence | feature_difference
camera_motion_compensation: true
context_token_fraction: 0.20
boundary_token_fraction: 0.20
boundary_band_seconds: 1.0
boundary_nms_seconds: 4.0
maximum_start_bands: 4
maximum_end_bands: 4
semvid_retention_ratio: 0.125
```

### Per-query output

Record:

- evidence backend and fallback reason;
- raw and camera-compensated motion summaries;
- start/end band peaks, widths, and scores;
- retained token counts by context, start, end, object, motion, and adaptive roles;
- number of frames receiving their minimum floor;
- dense and retained token totals;
- evidence mass before, inside, and after the prediction;
- Qwen interval, refined interval, objective gain, and rejection reason;
- vision, selection, prefill, generation, refinement, and end-to-end latency.

## Evaluation design

### Datasets

- TACoS is the primary fine-boundary and procedural-action benchmark.
- Charades-STA tests short indoor actions and repeated interactions.
- ActivityNet Grounding tests longer and visually diverse events.
- EGTEA-style gaze data may be used only for diagnostic saliency analysis, not training or main VTG ranking.

### Equal-token baselines

1. dense full-video Qwen;
2. official SemVID;
3. uniform token retention;
4. query-only selection;
5. raw-motion SemVID;
6. camera-compensated novelty without boundary roles;
7. boundary roles without camera compensation;
8. complete BEC-SemVID.

The SemVID and BEC-SemVID comparisons must have identical sampled frames, dense vision input, retention ratio, generation prompt, and refinement setting.

### Token budgets

- 6.25%;
- 12.5%;
- 25%.

The 12.5% comparison is the primary equal-token claim. The other ratios test whether boundary protection becomes more valuable as compression increases.

### Essential ablations

- raw versus camera-compensated motion;
- same-location versus motion-corresponded feature change;
- query, novelty, and multiplicatively gated boundary signals;
- no boundary pool versus start-only, end-only, and both;
- no global context floor versus 20% context;
- independent versus joint endpoint refinement;
- two versus four boundary bands;
- scene-cut anchors allowed versus query-gated.

### Metrics

- R@1 at IoU 0.3, 0.5, and 0.7;
- mIoU;
- start MAE and end MAE separately;
- boundary-band hit/coverage rate;
- semantic mass retained and boundary semantic mass retained;
- dense/retained tokens and prefill length;
- token-selection and end-to-end overhead;
- peak VRAM.

Use paired percentile-bootstrap confidence intervals clustered by video with 10,000 resamples and a declared fixed seed.

## Acceptance and falsification criteria

At 12.5% tokens, BEC-SemVID is supported if it satisfies either:

- at least three absolute points improvement in R@1@0.7; or
- at least 15% relative reduction in boundary MAE;

while also satisfying:

- no more than one absolute point loss in R@1@0.3;
- identical retained-token count to official SemVID;
- preprocessing overhead below 10% of total inference time;
- improvement on at least two of TACoS, Charades-STA, and ActivityNet.

A second successful outcome is matching official SemVID at 12.5% within two points using only 6.25% tokens with a meaningful prefill reduction.

The proposal is falsified if boundary quotas improve diagnostic token coverage but not endpoint accuracy, if global context loss reduces low-IoU recall, or if motion/codec processing costs more than it saves. In that case, CAM-SemVID should remain a diagnostic ablation rather than a claimed contribution.

## Scientific contribution and positioning

The proposed contribution is:

> A training-free, equal-token spatial allocation policy that preserves directional start/end evidence and the surrounding state contrast required for temporal grounding.

This is distinct from official [SemVID](https://arxiv.org/abs/2603.05663), whose object, motion, query, and context roles do not explicitly represent an ordered before/start/interior/end/after chain.

It imports an information principle from [Mage-VL](https://arxiv.org/abs/2607.24904)—codec predictability indicates redundant visual content—but not Mage-VL's trained sparse encoder. BEC-SemVID keeps Qwen's dense frozen visual front-end and uses codec evidence only after encoding. It also differs from learned interval representations such as [TimePLE](https://arxiv.org/abs/2607.23951) and learned evidence-pool approaches such as [F2G](https://arxiv.org/abs/2605.21973).

Human saliency remains a diagnostic direction described in [video_saliency_for_training_free_vtg.md](video_saliency_for_training_free_vtg.md). Free-viewing gaze is not assumed to equal query-conditioned relevance.

## Implementation order

1. reproduce official SemVID at 6.25%, 12.5%, and 25%;
2. implement camera-compensated novelty as an optional SemVID motion backend;
3. validate exact equal-token accounting and chronological reconstruction;
4. implement directional start/end evidence bands;
5. add context/boundary/adaptive quota allocation;
6. run equal-token token-retention diagnostics without refinement;
7. run full grounding comparisons;
8. add joint refinement only after the pruning effect is isolated;
9. execute full datasets after fixed-subset acceptance gates pass.

## Relationship to Proposal 1

[Budget-Conserving Semantic Corridor Routing](proposal_budget_conserving_semantic_corridor.md) and BEC-SemVID must first be evaluated independently:

1. BC-SCR uses unchanged official SemVID and must establish safe temporal savings.
2. BEC-SemVID uses continuous full-video input and must improve equal-token spatial grounding.
3. Only after both succeed should BC-SCR feed one corridor into BEC-SemVID.

This order separates temporal recall failure from spatial evidence loss and makes the final hybrid contribution scientifically interpretable.

## Related notes

- [Current progress report](current_progress_report.md)
- [Codec-assisted SemVID](first_spatial_improvement_codec_assisted_semvid.md)
- [Future boundary-evidence corridor](future_boundary_evidence_corridor.md)
- [Video saliency for training-free VTG](video_saliency_for_training_free_vtg.md)
- [Temporal failure diagnosis](temporal_sparsification_failure_note.md)
