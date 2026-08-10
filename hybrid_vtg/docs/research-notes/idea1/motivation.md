# Research Motivation — Training-Free Hierarchical Pruning for Long-Video Temporal Grounding

## Core motivation

Long videos contain two distinct forms of redundancy:

1. **Temporal redundancy:** most time intervals are unrelated to the query.
2. **Spatial redundancy:** even inside a relevant interval, most image patches depict background, static, repetitive, or query-irrelevant content.

A single pruning mechanism addresses only one of these problems. Q-MoG-TF removes redundancy hierarchically:

\[
\text{whole-video temporal filtering}
\rightarrow
\text{local spatial-token filtering}
\rightarrow
\text{temporal grounding and refinement}.
\]

This hierarchy matches the natural structure of video temporal grounding (VTG): first determine **where to look**, then determine **which visual evidence to preserve**. The pipeline uses a cheap low-FPS encoder for global temporal retrieval, followed by expensive high-FPS encoding and patch-level sparsification only inside selected regions.

---

## 1. Long-video VTG is a search problem before it is a localization problem

Consider the query:

> “When does the person open the refrigerator?”

The target may occupy only five seconds of a one-hour video. Processing the complete video at high resolution and high FPS is wasteful because

\[
\frac{\text{relevant duration}}{\text{video duration}} \ll 1.
\]

Before predicting precise boundaries, the system must locate the small region that may contain the answer. This motivates a coarse temporal stage that:

- samples the complete video cheaply;
- compares multi-scale temporal windows with the query;
- retains several high-relevance candidates;
- discards clearly irrelevant periods.

The first stage solves **global search**. The later stages solve **local recognition and localization**.

---

## 2. Temporal pruning alone is insufficient

Suppose temporal filtering reduces a one-hour video to two candidate windows totaling three minutes. This is a substantial reduction, but the retained frames may still contain enormous spatial redundancy:

- walls, floors, sky, and furniture;
- static background;
- repeated patches across adjacent frames;
- people or objects unrelated to the query.

A vision encoder may generate hundreds of patch tokens per frame. At a moderate FPS, even a short candidate window can produce tens of thousands of visual tokens. Therefore,

\[
\text{small temporal duration}
\not\Rightarrow
\text{small visual-token sequence}.
\]

Temporal filtering reduces the number of frames, but not the redundancy inside each retained frame. This motivates the spatial stage, which scores patches using query relevance, motion, visual uniqueness, and boundary importance before retaining or merging them.

---

## 3. Spatial-token pruning alone is also insufficient

Applying patch pruning to the complete video still requires:

- decoding frames across the entire timeline;
- passing all sampled frames through the vision encoder;
- calculating patch features for irrelevant periods;
- computing pruning scores across the full video.

Even if 90% of post-encoder tokens are removed, most of the expensive vision-encoding cost may already have been paid:

\[
\text{post-encoder token reduction}
\neq
\text{full-pipeline compute reduction}.
\]

The coarse temporal stage prevents irrelevant frames from reaching the expensive expert encoder. The spatial stage then reduces the remaining projector, language-model, and grounding costs. They act on different terms in the total computation:

\[
C_{\text{total}}
=
C_{\text{coarse search}}
+C_{\text{expert vision}}
+C_{\text{token processing}}
+C_{\text{grounding}}
+C_{\text{refinement}}.
\]

- Temporal pruning primarily reduces \(C_{\text{expert vision}}\).
- Post-encoder spatial pruning primarily reduces \(C_{\text{token processing}}\) and \(C_{\text{grounding}}\).
- Boundary refinement adds a small, explicitly measured \(C_{\text{refinement}}\).

This distinction is essential for honest efficiency claims: post-encoder token pruning does not retroactively reduce vision-tower computation.

---

## 4. Each stage operates at an appropriate semantic resolution

Coarse temporal retrieval does not require detailed patch-level reasoning. A low-resolution global representation may be sufficient to reject a driving scene when the query concerns cooking.

Once a plausible window has been selected, however, precise grounding may depend on fine details:

- a hand touching an object;
- a door beginning to move;
- an object entering or leaving the scene;
- a visual state transition;
- a particular person performing an action.

This creates a natural allocation of model capacity:

| Stage | Resolution | Purpose |
|---|---|---|
| Coarse temporal retrieval | Low FPS, global features | Find likely temporal regions |
| Candidate-region processing | Higher FPS | Recover local event dynamics |
| Spatial sparsification | Patch-level features | Preserve query-relevant evidence |
| Boundary refinement | Dense local sampling | Estimate precise start and end times |

The hybrid architecture applies expensive reasoning only where fine detail is useful.

---

## 5. Temporal and spatial relevance are complementary

A temporal window can be globally relevant while most of its patches remain irrelevant. Conversely, a frame can have modest global similarity while containing a small query-critical object.

Consider:

> “When does the person place the red cup on the table?”

A window may be retained because it contains the person and the table. Within that window:

- the cup patches provide query-specific evidence;
- the person’s hand provides motion evidence;
- the table provides static contextual evidence;
- frames around contact provide boundary evidence;
- unrelated background patches are redundant.

The temporal stage asks:

> “Which interval probably contains the event?”

The spatial stage asks:

> “Which visual elements explain the event and its boundaries?”

These are related but non-identical decisions.

---

## 6. Motion becomes useful after semantic narrowing

Motion alone is not a reliable global search signal. A long video may contain substantial irrelevant motion from:

- camera movement;
- other people walking;
- passing vehicles;
- lighting changes;
- unrelated scene transitions.

Applied to the complete video, motion pruning may preserve dynamic but query-irrelevant content. After query-guided temporal filtering, motion becomes more meaningful because it is evaluated only inside semantically plausible windows:

\[
\text{useful motion}
\approx
\text{motion conditioned on a query-relevant temporal region}.
\]

The hierarchy reduces the risk that generic movement dominates the pruning decision.

---

## 7. Query relevance alone cannot preserve event boundaries

The highest query similarity often occurs near the center of an event rather than at its start or end:

- “opens the door” may score highest when the door is already open;
- “sits down” may score highest after the person is seated;
- “picks up the cup” may score highest while the cup is already being held.

Temporal grounding requires the complete transition:

\[
\text{before event}
\rightarrow
\text{event begins}
\rightarrow
\text{event occurs}
\rightarrow
\text{event ends}.
\]

This motivates:

- temporal halos around selected windows;
- motion and feature-change evidence;
- explicit retention of boundary-important tokens;
- high-FPS local boundary refinement.

The method is therefore not merely a compression pipeline. It must preserve the **evidence chain** needed to distinguish event occurrence from event boundaries.

---

## 8. Hybrid pruning enables conditional computation

Different query–video pairs require different amounts of computation. Some queries produce a concentrated temporal similarity distribution:

\[
s_1 \gg s_2,s_3,\ldots
\]

and may require only one or two windows. Ambiguous or repeated events produce diffuse scores and require broader temporal coverage.

A hierarchical system can allocate computation conditionally:

- retain more temporal windows when retrieval confidence is low;
- allocate more frames near likely transitions;
- retain more spatial tokens in highly relevant or boundary-rich frames;
- retain fewer tokens in repetitive or static frames.

This is more efficient than assigning every query the same sampling rate and token budget. In the training-free setting, every allocation rule must remain deterministic, label-independent, and fully recorded.

---

## 9. Hierarchical pruning makes aggressive compression safer

One extremely aggressive pruning operation creates a high risk of catastrophic information loss:

- retaining only 2% of frames may remove the target interval entirely;
- retaining only 2% of patches may remove a small but essential object.

A two-stage design distributes compression:

\[
r_{\text{total}}
=
r_{\text{temporal}}
\times
r_{\text{spatial}}.
\]

For example,

\[
0.15\times0.25=0.0375.
\]

The system retains only 3.75% of the original visual-token volume, although neither stage individually operates at an extreme 3.75% budget. This may provide a safer accuracy–efficiency trade-off:

- temporal filtering preserves candidate coverage;
- spatial filtering preserves local semantic and boundary evidence;
- their product provides strong total compression.

Temporal, expert-frame, spatial-token, and total-compute budgets must therefore be measured separately rather than collapsed into one pruning ratio.

---

## 10. The hierarchy is compatible with fully training-free inference

A training-free system cannot learn a new pruning policy from VTG annotations. It must organize reusable signals from frozen pretrained models and deterministic visual analysis:

- text–video similarity for temporal retrieval;
- patch–query similarity for semantic relevance;
- motion-compensated feature change for dynamics;
- token diversity for uniqueness and redundancy;
- visual continuity and evidence change for boundary refinement.

These signals naturally operate at different temporal and spatial resolutions. Combining them hierarchically is more plausible than expecting one heuristic to solve global retrieval, local evidence selection, and boundary localization simultaneously.

The training-free argument is:

> Frozen pretrained models already contain useful semantic and visual information, but processing every frame and patch is unnecessary. A deterministic hierarchy can expose only the most useful evidence to a frozen grounding model.

The contribution is not a newly fitted representation. It is the efficient organization of pretrained evidence without SFT, GRPO, optimization, or parameter updates.

---

## 11. The hybrid method improves interpretability

Every stage produces an observable intermediate result:

1. temporal candidate windows and their scores;
2. retained connected components and detailed frames;
3. per-token query, motion, uniqueness, and boundary scores;
4. retained or merged spatial patches;
5. coarse and refined temporal intervals.

This makes failure attribution possible:

- Did temporal retrieval discard the correct region?
- Did spatial pruning remove a query-critical object?
- Did the method preserve the event center but lose a boundary?
- Did the frozen VideoLLM fail despite receiving sufficient evidence?

The system can therefore evaluate candidate recall, frame coverage, token preservation, grounding quality, and boundary error separately. This diagnostic clarity is a scientific benefit over a monolithic pipeline.

---

## 12. Central research hypothesis

> Evidence for video temporal grounding is sparse at multiple scales. Query-relevant events occupy a small fraction of a long video, and the visual evidence needed to recognize and localize those events occupies a small fraction of the retained frames. Therefore, hierarchical temporal and spatial sparsification should reduce computation more effectively than either temporal or spatial pruning alone, while preserving grounding accuracy through explicit protection of semantic, motion, contextual, and boundary evidence.

A concise version is:

> **First determine where the event may occur, then preserve only the evidence needed to identify what happens and where its boundaries lie.**

---

## 13. Why not use only one stage?

| Method | Main weakness |
|---|---|
| Temporal pruning only | Candidate windows still produce many redundant patch tokens |
| Spatial pruning only | Irrelevant frames must still be decoded and encoded |
| Motion-only pruning | Preserves irrelevant motion and removes static but relevant evidence |
| Query-only pruning | May preserve event centers while losing transitions and boundaries |
| Uniform sampling | Ignores query relevance and event duration |
| One global token selector | Must process all tokens and may allocate budget poorly across time |
| **Hierarchical hybrid method** | Separates global search from local evidence preservation |

---

## 14. Honest limitations

### 14.1 Cascaded error

If temporal retrieval removes the correct interval, later stages cannot recover it. The temporal stage must therefore prioritize candidate recall and use halos and uncertainty-aware coverage rather than optimizing compression alone.

### 14.2 Increased system complexity

The hierarchy introduces several inference hyperparameters:

- temporal window sizes and overlap;
- frame and temporal-union budgets;
- halo duration;
- coarse and detailed sampling rates;
- spatial-token quotas and score weights;
- boundary-refinement radius.

A training-free study must not tune these independently on each test benchmark. The complete contract must be fixed in advance, and all attempted development settings must be disclosed.

### 14.3 Potential overhead

Temporal retrieval, motion compensation, token scoring, merging, and boundary refinement are not free. The method is useful only if their overhead is smaller than the vision and language-model computation they eliminate. End-to-end latency, memory, and preprocessing cost must be measured directly.

### 14.4 Static evidence

Motion-based pruning can remove stationary but semantically important objects, states, or context. Query relevance, visual uniqueness, spatial coverage, and protected context quotas must therefore complement motion.

### 14.5 Frozen-model compatibility

The current TimeLens2 inference wrapper consumes frames rather than externally selected visual embeddings. Sparse-token injection must preserve placeholder counts, position information, and any deep-stack visual features. Until 100% token retention reproduces the unmodified frozen model, token-level VideoLLM savings remain an engineering hypothesis rather than an established result.

---

## Paper-style motivation

> Long-video temporal grounding contains structured redundancy along both temporal and spatial dimensions. The queried event often occupies only a small fraction of the complete video, while the visual evidence required to recognize and localize that event occupies only a small subset of patches within the relevant interval. Temporal-only filtering continues to process substantial intra-frame redundancy, whereas spatial-only pruning still incurs the cost of decoding and encoding irrelevant temporal regions. We therefore adopt a hierarchical hybrid paradigm: inexpensive query-guided temporal search first restricts expensive processing to high-recall candidate intervals, after which query-, motion-, uniqueness-, and boundary-aware token sparsification preserves the local evidence required for precise grounding. This organization naturally supports training-free inference because every decision is derived from frozen pretrained representations and deterministic visual signals without task-specific parameter updates. The resulting method first determines where the event may occur, then preserves only the evidence needed to identify the event and localize its boundaries.
