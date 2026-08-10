# Motivation and Research Rationale for Q-MoG-TF

## 1. Research premise

Video temporal grounding (VTG) asks a model to identify the start and end times of the video interval described by a natural-language query. The task becomes difficult in long videos because the amount of visual data grows with video duration, while the answer usually remains short. For example, the action *“the person opens the refrigerator”* may last only a few seconds in an hour-long recording.

This creates a severe mismatch between the amount of video that can be processed and the amount of video that is actually useful:

\[
\frac{\text{duration of relevant evidence}}
     {\text{total video duration}}
\ll 1.
\]

The same imbalance also exists within a relevant frame. A frame may contain hundreds of visual patch tokens, even though only a hand, an object, and a small amount of surrounding context are necessary to recognize the queried event. The core premise of this research is therefore that evidence for long-video temporal grounding is sparse at two distinct levels:

1. **Temporal sparsity:** only a small part of the video is related to the query.
2. **Spatial/token sparsity:** only a small part of each relevant frame is needed to recognize the event and locate its boundaries.

These two forms of redundancy arise at different points in the inference pipeline and cannot be removed effectively by a single pruning operation. Q-MoG-TF consequently uses a hierarchy:

\[
\text{whole-video temporal search}
\rightarrow
\text{local high-resolution processing}
\rightarrow
\text{spatial-token sparsification}
\rightarrow
\text{temporal grounding and boundary refinement}.
\]

The guiding principle is simple:

> First determine where the event may occur; then preserve only the local evidence needed to determine what happens and where it begins and ends.

---

## 2. Long-video grounding is a search problem before it is a localization problem

A model cannot precisely localize an event that it has not first found. In a long video, exhaustive high-frame-rate processing spends most of its computation on intervals that have no relationship to the query. The first problem is therefore global search: identify a small, high-recall set of temporal regions that may contain the answer.

This observation motivates a cheap whole-video scan. Q-MoG-TF samples the complete video at low frame rate and low spatial resolution, extracts reusable frozen features, compares multi-scale temporal windows with the query, and retains a diverse set of plausible candidates. Only the selected temporal union is decoded and encoded at higher frame rate.

The coarse stage is not expected to determine exact boundaries. Its role is deliberately narrower:

- reject clearly irrelevant portions of the video;
- preserve short events through peak as well as average similarity;
- cover events of different duration through multi-scale windows;
- maintain high candidate recall through multiple candidates, temporal halos, and a low-confidence coverage fallback.

The distinction between search and localization is essential. The temporal retriever should be judged primarily by whether the target and its boundaries survive into the candidate set, not by whether its top-ranked window already matches the final annotation exactly.

---

## 3. Why temporal pruning alone is insufficient

Temporal filtering can reduce one hour of video to a few candidate minutes, but those minutes may still produce an enormous visual sequence. Each retained frame can contain hundreds of patch tokens representing walls, floors, furniture, sky, static background, repeated texture, and unrelated people or objects. Similar patches are also repeated across adjacent frames.

Consequently,

\[
\text{short retained duration}
\not\Rightarrow
\text{short visual-token sequence}.
\]

Temporal pruning reduces the number of frames reaching the expensive vision tower. It does not, by itself, remove redundancy within the retained frames. Passing every patch from every candidate frame to the multimodal language model still incurs substantial projector, attention, memory, and grounding costs.

This motivates a second stage operating at spatial-token resolution. Within the selected intervals, Q-MoG-TF scores visual tokens using four complementary signals:

- **query relevance**, to preserve objects and regions semantically related to the text;
- **motion-compensated change**, to preserve local action evidence while discounting camera motion;
- **visual uniqueness**, to protect static but distinctive evidence that neighboring tokens cannot represent;
- **boundary importance**, to preserve changes associated with the beginning and end of an event.

Protected quotas ensure that one strong signal cannot consume the complete budget and erase another type of evidence. The remaining budget is filled by a deterministic combined score. Spatial sparsification therefore addresses the redundancy that remains after temporal search.

---

## 4. Why spatial pruning alone is insufficient

Applying token pruning to uniformly sampled frames from the complete video does not solve the main cost of long-video processing. Before post-encoder pruning can score a patch, the system normally must still:

- decode the corresponding frame;
- resize and preprocess it;
- execute the expensive vision encoder;
- construct patch representations for query-irrelevant periods.

Even if most tokens are discarded afterward, much of the upstream visual cost has already been paid:

\[
\text{post-encoder token reduction}
\neq
\text{end-to-end compute reduction}.
\]

Temporal and spatial pruning therefore reduce different terms in the computational budget. A useful decomposition is

\[
C_{\mathrm{total}}
=
C_{\mathrm{scan}}
+C_{\mathrm{retrieval}}
+C_{\mathrm{expert\ vision}}
+C_{\mathrm{spatial\ scoring}}
+C_{\mathrm{LLM/grounding}}
+C_{\mathrm{refinement}}.
\]

Temporal selection primarily reduces decoding and expert-vision cost by preventing irrelevant frames from reaching the expensive encoder. Spatial selection primarily reduces the visual sequence passed to the projector or language model, thereby lowering downstream memory, attention, and grounding cost. The research must measure these terms separately: a token-retention ratio cannot be treated as an end-to-end speedup.

---

## 5. Why the stages use different semantic resolutions

Global rejection and precise localization do not require the same visual detail. A low-resolution global embedding may be sufficient to reject a window of driving when the query concerns cooking. Precise grounding, however, may depend on a small hand movement, an object changing state, a door beginning to move, or an object making contact with a surface.

This leads to a natural allocation of representational capacity:

| Stage | Resolution | Main purpose |
|---|---|---|
| Whole-video scan | Low FPS, low resolution, global frozen features | Search the complete timeline cheaply |
| Temporal retrieval | Multi-scale window features | Preserve a high-recall candidate set |
| Candidate processing | Higher FPS and expert visual features | Recover local event dynamics and detail |
| Spatial sparsification | Patch/token features | Retain semantic, motion, contextual, and transition evidence |
| Boundary refinement | Dense sampling near two predicted endpoints | Improve start and end precision |

The hierarchy assigns expensive computation only when additional resolution can change the answer. It also allows the same coarse video index to be reused across multiple queries, separating one-time indexing cost from amortized per-query cost.

---

## 6. Temporal relevance and spatial relevance are complementary

A temporally relevant window is not uniformly relevant in space. Conversely, a frame with moderate global similarity can contain one small but decisive object. Consider the query *“When does the person place the red cup on the table?”* A selected interval may contain the correct person and table, yet different regions play different roles:

- the cup provides object identity;
- the hand provides action and motion evidence;
- the table provides contextual and final-state evidence;
- the contact transition helps determine the event boundary;
- most of the remaining background is unnecessary.

The temporal stage answers, *“Which intervals may contain the event?”* The spatial stage answers, *“Which visual elements within those intervals constitute sufficient evidence?”* These decisions are related but not interchangeable.

This complementarity is also why a single global token selector is unattractive. It would have to compare tokens from the entire video, after much of the computational cost had already been incurred, and could spend the budget disproportionately on long or visually salient but irrelevant periods. Temporal restriction gives spatial selection a semantically plausible local domain in which its finer signals become meaningful.

---

## 7. Why motion is conditioned on semantic narrowing

Motion is informative for action boundaries, but it is a poor whole-video retrieval signal on its own. Long videos contain abundant query-irrelevant motion caused by camera movement, walking people, vehicles, lighting changes, and shot transitions. A global motion policy may therefore preserve dynamic distractors while discarding a static object or final state that is crucial to the query.

After query-guided temporal filtering, motion has a more useful interpretation:

\[
\text{useful motion}
\approx
\text{motion within a semantically plausible interval}.
\]

Q-MoG-TF further estimates and compensates for global camera motion before measuring residual local change. Motion is then combined with query relevance and uniqueness, rather than treated as a sufficient criterion. The order of the hierarchy is important: semantic narrowing makes local motion evidence less likely to be dominated by irrelevant activity elsewhere in the video.

---

## 8. Why query similarity alone cannot preserve event boundaries

The frame that most strongly resembles a query often lies near the semantic center of an event rather than at its start or end. For *“opens the door,”* similarity may peak when the door is already open. For *“sits down,”* it may peak after the person is seated. For *“picks up the cup,”* it may peak while the cup is already being held.

Temporal grounding, however, requires evidence for a transition:

\[
\text{pre-event state}
\rightarrow
\text{event onset}
\rightarrow
\text{event interior}
\rightarrow
\text{event offset}
\rightarrow
\text{post-event state}.
\]

A system that retains only the most query-similar frames may recognize that the event occurred but still fail to localize when it began and ended. Q-MoG-TF protects this evidence chain in three places:

1. temporal halos retain context immediately outside selected windows;
2. boundary-aware token scoring preserves frames and patches near changes in query evidence and visual state;
3. high-FPS local refinement compares evidence on both sides of each predicted endpoint.

The method is therefore not merely a compression system. It is a task-aware evidence-preservation system whose compression policy is designed around the requirements of temporal localization.

---

## 9. Conditional computation and distributed compression

Different query-video pairs do not require the same amount of computation. Some queries produce one sharply dominant temporal region; others describe repeated or ambiguous events and yield a diffuse relevance distribution. A hierarchical design can respond conditionally:

- retain more temporal coverage when retrieval confidence is low;
- preserve multiple diverse candidates when the event may repeat;
- assign more frame or token budget to strongly relevant or transition-heavy regions;
- compress repetitive and static background more aggressively;
- spend dense sampling only near the two predicted boundaries.

The hierarchy also makes aggressive total compression safer. If temporal and spatial retention ratios are $r_t$ and $r_s$, respectively, then the approximate retained visual-token fraction is

\[
r_{\mathrm{total}} = r_t r_s.
\]

For example,

\[
0.15\times0.25=0.0375.
\]

The system retains approximately $3.75\%$ of the original visual-token volume, even though neither stage individually uses an extreme $3.75\%$ selection rate. Compression is distributed across two safer decisions: temporal filtering preserves candidate coverage, and spatial filtering preserves diverse local evidence. This is the central reason to expect a better accuracy-efficiency trade-off than either stage alone.

---

## 10. Why a training-free formulation is scientifically useful

Q-MoG-TF asks whether the representations already present in frozen pretrained models are sufficient for efficient grounding when their evidence is organized appropriately. It introduces no task-specific optimizer, projection, router, grounding head, or token gate. Instead, it derives every decision from frozen or analytic signals:

- frozen text-video similarity for temporal retrieval;
- frozen token-query similarity for semantic evidence;
- motion-compensated feature change for action evidence;
- feature diversity for static uniqueness;
- changes in relevance and visual continuity for boundary evidence;
- deterministic thresholds, budgets, quotas, and tie-breaking rules.

This restriction serves several purposes. It isolates the contribution of inference-time evidence allocation from gains due to additional training data or parameters. It permits use with existing checkpoints and avoids the expense of retraining a long-video model. It also makes intermediate decisions auditable and allows the same policy to be evaluated across benchmarks under a frozen experimental contract.

The training-free claim should remain precise. Q-MoG-TF may use publicly pretrained encoders and grounders, but it performs no task-specific parameter updates. Ground-truth intervals are visible only to the evaluator and oracle diagnostics, never to routing, pruning, grounding, or refinement.

---

## 11. Interpretability and scientific diagnosis

The hierarchy exposes an observable evidence trail:

1. all temporal candidates and their relevance scores;
2. the selected windows, halos, and connected components;
3. the detailed frames sent to the expert encoder;
4. query, motion, uniqueness, and boundary scores for each token;
5. protected quotas and final retention masks;
6. the coarse interval and each boundary-refinement decision;
7. the final interval in absolute video time.

This decomposition distinguishes several failure modes that a monolithic prediction obscures:

- **search failure:** the target interval never enters the candidate set;
- **coverage failure:** the event center is retained but a start or end boundary is truncated;
- **representation failure:** a relevant object, state, or transition token is removed;
- **reasoning failure:** the frozen grounder receives sufficient evidence but predicts the wrong interval;
- **refinement failure:** a plausible coarse interval is adjusted toward the wrong transition.

The evaluation can therefore report candidate recall, start/end coverage, retained-frame and token statistics, boundary error, grounding accuracy, and component-wise latency. This is not only an engineering convenience; it makes the causal claims of the research testable.

---

## 12. Central research question and hypotheses

The main research question is:

> Can a deterministic, training-free hierarchy jointly remove temporal and spatial redundancy in long videos while preserving the semantic and transition evidence required for accurate temporal grounding?

The central hypothesis is:

> Evidence for long-video temporal grounding is sparse at multiple scales. Query-relevant events occupy a small portion of a long video, and the evidence needed to identify those events occupies a small portion of the retained frames. Hierarchical temporal and spatial sparsification should therefore reduce computation more effectively than temporal-only or spatial-only pruning, provided that the hierarchy explicitly protects candidate recall, static contextual evidence, motion evidence, and event boundaries.

This hypothesis yields four testable sub-hypotheses:

1. **Temporal-search hypothesis:** multi-scale query-guided retrieval can remove most irrelevant duration while maintaining high candidate and boundary recall.
2. **Spatial-sparsity hypothesis:** within fixed temporal routes, query-, motion-, uniqueness-, and boundary-aware token selection can improve the quality-token or quality-latency frontier over dense tokens and simpler pruning rules.
3. **Boundary-preservation hypothesis:** temporal halos, explicit boundary evidence, and local high-FPS refinement improve high-tIoU recall or start/end error at a matched compute budget.
4. **Hybrid-complementarity hypothesis:** temporal and spatial sparsification together produce a better end-to-end accuracy-efficiency frontier than either temporal pruning on dense tokens or spatial pruning on uniformly sampled full-video frames.

The strongest version of the research claim is not that fewer tokens are processed. It is that the full hybrid method achieves a measurably better grounding-quality versus wall-time, memory, or downstream-token frontier under strict, fully charged budgets.

---

## 13. Why common alternatives are incomplete

| Alternative | Fundamental weakness for long-video VTG |
|---|---|
| Temporal pruning only | Selected intervals still contain large amounts of redundant spatial-token content. |
| Spatial pruning only | Irrelevant frames must still be decoded and visually encoded before post-encoder pruning. |
| Motion-only pruning | Dynamic but query-irrelevant content dominates, while static relevant evidence may disappear. |
| Query-only pruning | Event centers may survive while state transitions and precise boundaries are lost. |
| Uniform sampling | The budget is unrelated to query relevance, event duration, repetition, or uncertainty. |
| One global token selector | It pays global feature cost and may allocate the token budget poorly across time. |
| Naive temporal-plus-spatial cascade | Fixed independent thresholds can compound errors without protecting recall or boundaries. |
| Q-MoG-TF hierarchy | Separates global search from local evidence preservation and explicitly protects complementary evidence types. |

The relevant baseline is not merely a dense model. The experiments must compare against temporal-only, spatial-only, random-token, query-only, motion-only, and naive combined policies at matched frame and token budgets. Otherwise, the contribution of the hierarchy cannot be isolated.

---

## 14. Risks and limitations

### Cascaded error

The primary risk is irreversible temporal failure. If the first stage removes the true interval, no later spatial policy or grounder can recover it. Temporal selection must therefore optimize high candidate recall rather than aggressive Top-1 precision. Multi-scale windows, diversity, halos, uncertainty fallback, and explicit recall diagnostics mitigate this risk but cannot eliminate it.

### Small or globally inconspicuous evidence

A tiny query-critical object may have weak influence on a global window embedding. Peak frame similarity and multiple window scales help, but extremely small, brief, or occluded evidence may still be missed before expert encoding.

### Static evidence

Motion is not equivalent to relevance. Stationary objects, subtitles, scene context, and final states can define the correct answer. Query relevance, visual uniqueness, contextual coverage, and minimum per-frame quotas are therefore necessary companions to motion.

### Added system complexity

The hierarchy introduces window lengths, strides, halos, frame rates, score weights, token quotas, refinement radii, and confidence rules. A training-free paper must freeze these choices before test evaluation, report declared development sweeps, and avoid benchmark-specific test tuning.

### Overhead may exceed savings

Low-FPS retrieval, camera-motion estimation, token scoring, merging, and boundary refinement are not free. The method is useful only if their measured overhead is smaller than the decoding, expert-vision, language-model, and memory costs they remove. Retention percentages alone are insufficient evidence.

### Frozen-model interface constraints

Post-encoder pruning requires safe access to visual embeddings, placeholder positions, attention masks, and positional information. If sparse injection changes the behavior of the 100%-retention path or violates model invariants, the system cannot claim valid sparse VideoLLM inference. The analytic proposal scorer remains a transparent fallback and control.

### No guaranteed generalization from heuristics

Deterministic frozen signals improve reproducibility but cannot adapt their semantics through task-specific learning. Their fixed weights may behave differently across video styles, event durations, and benchmarks. Cross-dataset evaluation and leave-one-signal-out ablations are therefore necessary.

---

## 15. Logical chain from problem to method

The complete reasoning behind the research can be summarized as follows:

1. Long videos are expensive because dense visual processing scales with duration and visual-token count.
2. Ground-truth events occupy only a small temporal fraction of most long videos.
3. The visual evidence needed inside a relevant interval occupies only a small spatial/token fraction of its frames.
4. Temporal pruning saves upstream decoding and expert-encoder work but leaves downstream token redundancy.
5. Spatial pruning saves downstream token processing but, by itself, pays the cost of encoding irrelevant time periods.
6. A temporal-to-spatial hierarchy removes each kind of redundancy at the point where it arises.
7. Query similarity identifies semantic evidence but does not reliably preserve event transitions.
8. Motion identifies change but is noisy globally and can discard static evidence.
9. Uniqueness, contextual quotas, temporal halos, and boundary-aware scoring make the hierarchy safer.
10. High-FPS refinement is most economical near the two endpoints, where additional temporal resolution directly affects localization accuracy.
11. Frozen pretrained features provide all required signals, allowing the hierarchy to be evaluated without task-specific training.
12. The research succeeds only if the complete system improves the measured accuracy-efficiency Pareto frontier under strict, component-wise accounting.

---

## 16. Paper-ready motivation

Long-video temporal grounding contains structured redundancy along both temporal and spatial dimensions. A queried event often occupies only a small fraction of the complete video, while the visual evidence needed to recognize and localize that event occupies only a small subset of patches within the relevant interval. Temporal-only filtering still passes substantial intra-frame redundancy to the multimodal grounder, whereas post-encoder spatial pruning alone continues to incur the cost of decoding and encoding irrelevant temporal regions. We therefore study a hierarchical, training-free paradigm in which an inexpensive query-guided scan first restricts expensive processing to a high-recall set of candidate intervals, after which query-, motion-, uniqueness-, and boundary-aware token sparsification preserves the local evidence required for precise grounding. Temporal halos and high-frame-rate endpoint refinement protect the pre- and post-event transitions that semantic similarity alone tends to miss. This organization uses frozen pretrained representations and deterministic allocation rules, enabling computation to be concentrated where it is most informative without task-specific parameter updates. Our central hypothesis is that temporally locating plausible evidence before spatially compressing it will yield a better grounding-accuracy versus compute frontier than temporal pruning, spatial pruning, or uniform sampling alone.

## 17. One-sentence thesis

> Q-MoG-TF treats efficient long-video temporal grounding as hierarchical evidence preservation: search broadly and cheaply for where an event may occur, inspect those regions in detail, retain only the semantic and transition evidence needed to explain the event, and spend dense computation only on its boundaries.
