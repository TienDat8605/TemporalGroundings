# Q-MoG: Query-Conditioned Motion and Boundary-Aware Pruning for Efficient Long-Video Temporal Grounding

## Formal Research Proposal

### Abstract

Natural-language temporal grounding (VTG) requires a model to identify the start and end times of the video segment described by a text query. Long videos make this task computationally difficult: exhaustive visual encoding is expensive, and passing all visual tokens to a cross-modal transformer or multimodal large language model (MLLM) creates prohibitive attention and memory costs. Existing work addresses parts of this problem through query-guided temporal retrieval, hierarchical temporal search, sidekick/expert encoding, or generic visual-token pruning. However, treating temporal retrieval and spatial pruning as independent compression stages creates a task-specific failure mode: an early decision can remove the true event boundary or static query-relevant evidence, making accurate localization impossible.

This project proposes **Q-MoG**, a query-conditioned, boundary-preserving framework that jointly allocates computation across temporal windows, frames, and spatial tokens. Q-MoG formulates efficient VTG as a constrained evidence-selection problem. A reusable coarse video index supports multi-scale, high-recall temporal retrieval; a query-conditioned allocator assigns expert-encoder and token budgets using semantic relevance, motion residuals, visual uniqueness, boundary likelihood, and predictive uncertainty; a sparse grounding network predicts start and end distributions; and a high-frame-rate local refiner resolves the two predicted boundaries. Training combines span supervision, retrieval ranking, boundary coverage, dense-teacher distillation, and an explicit compute penalty. The primary hypothesis is that grounding-aware allocation can retain approximately 2–6% of the original downstream visual tokens while improving the accuracy–latency Pareto frontier over temporal-only retrieval, token-only pruning, and their naive composition.

The intended contribution is not the mere combination of temporal filtering and token pruning. It is the study and design of **joint, boundary-aware evidence selection under a global compute budget**, with explicit protection against irreversible retrieval failure.

---

## 1. Motivation and problem statement

Let a video be $V=\{f_t\}_{t=1}^{T}$, with duration $D$, and let $q$ be a natural-language query referring to a ground-truth temporal interval

\[
y^*=(t_s^*,t_e^*), \qquad 0\le t_s^*\le t_e^*\le D.
\]

The objective is to predict

\[
\hat y=(\hat t_s,\hat t_e)
\]

while minimizing both localization error and computation. A dense system processes $F$ frames with $P$ patch tokens per frame, producing $N=FP$ visual tokens. Long videos increase the cost of visual encoding approximately linearly in $F$, while downstream attention can grow linearly or quadratically with sequence length depending on the architecture and attention pattern.

The proposed research studies the constrained objective

\[
\min_{\theta,\,A}
\;\mathbb E_{(V,q,y^*)}
\left[
\mathcal L_{\mathrm{VTG}}
+\lambda_C C(A;V,q)
\right]
\]

subject to

\[
\Pr\!\left(y^*\cap \mathcal C_A(V,q)\neq\varnothing\right)\ge \tau,
\]

where $A$ is the learned allocation policy, $C$ is measured or estimated compute, $\mathcal C_A$ is the selected temporal candidate set, and $\tau$ is a target candidate-recall level. This constraint captures the asymmetry of the task: removing irrelevant evidence is inexpensive, but removing the target interval is catastrophic.

### Research question

> How should a fixed visual-compute budget be allocated jointly across temporal windows, frames, and spatial tokens so that long-video grounding preserves query-relevant evidence and event boundaries?

### Central hypothesis

> Evidence required for long-video temporal grounding is sparse in time and space, but it cannot be selected reliably by temporal relevance or motion alone. Joint modeling of query relevance, motion-compensated change, static uniqueness, boundary likelihood, and uncertainty will preserve localization-critical evidence more effectively than independent temporal and token pruning at the same compute budget.

---

## 2. Prior work and research gap

### 2.1 Long-video temporal search

[CONE](https://arxiv.org/abs/2209.10918) applies query-guided sliding-window selection before fine grounding and reports substantial inference acceleration on Ego4D-NLQ and MAD. [DeCafNet](https://arxiv.org/abs/2505.16376) separates inexpensive sidekick processing from selective expert encoding. [T*](https://arxiv.org/abs/2504.02259) reframes long-video search as adaptive temporal and spatial zooming. More recent systems such as [TimeSearch](https://arxiv.org/abs/2504.01407) and Q-Fold further demonstrate hierarchical query-aware search and heterogeneous allocation of representational detail.

These methods establish that coarse-to-fine temporal selection is effective. They also expose an upper-bound problem: once the relevant region is excluded, the final grounder cannot recover it. The recent [ExtremeWhen analysis](https://arxiv.org/abs/2606.12300) explicitly decomposes hour-long grounding into search and localization, strengthening the case for measuring candidate recall separately from final localization.

### 2.2 Efficient visual-token processing

[PruneVid](https://aclanthology.org/2025.findings-acl.1024/) merges temporally static and spatially similar tokens and applies query-dependent pruning, removing more than 80% of visual tokens while retaining competitive performance on its evaluated video-understanding tasks. [MoPrune](https://aclanthology.org/2026.findings-acl.344/) uses scene structure, frame uniqueness, visual distinctiveness, and motion salience; its reported results show strong retention of dense-model performance at low token budgets. [QTSplus](https://arxiv.org/abs/2511.11910) already provides query-aware token scoring and instance-adaptive retention.

Therefore, query-aware scoring, motion-aware pruning, static-token preservation, and adaptive budgets are not individually sufficient novelty claims.

### 2.3 Gap

Existing approaches predominantly optimize one of three objectives:

1. finding relevant temporal regions;
2. compressing generic video representations; or
3. improving long-video question answering under a visual budget.

Temporal grounding differs from generic understanding because success depends not only on retaining semantic evidence from the interior of an event, but also on retaining evidence for the transition into and out of it. A relevance-driven retriever may select the event center while truncating its edges. A motion-driven pruner may preserve unrelated camera motion while deleting a static object, subtitle, or final state that defines the answer. Independent stages also lack a mechanism to trade temporal coverage against spatial detail under one global budget.

The missing research problem is thus:

> **Grounding-aware joint allocation with explicit candidate-recall and boundary-preservation objectives.**

---

## 3. Proposed novelty and contributions

The following claims are deliberately scoped as proposed contributions rather than unsupported “first” claims. They must be rechecked against contemporaneous work before submission.

### Contribution 1: A unified budgeted formulation of efficient VTG

Q-MoG will formulate temporal-window selection, expert frame encoding, spatial token retention, and boundary refinement as levels of one allocation problem. Unlike a sequential composition with fixed per-stage retention ratios, the system will optimize a global cost budget:

\[
C_{\mathrm{total}}=
C_{\mathrm{index}}+
C_{\mathrm{retrieval}}+
C_{\mathrm{expert}}+
C_{\mathrm{projector}}+
C_{\mathrm{grounder}}+
C_{\mathrm{refine}}.
\]

This formulation makes it possible to spend more computation on uncertain windows and probable boundaries while aggressively compressing easy background regions.

### Contribution 2: Boundary-preserving hierarchical evidence selection

Q-MoG will introduce explicit boundary sufficiency into both temporal and token selection. It will combine temporal halos, supervised boundary likelihood, boundary-neighborhood coverage loss, and local resampling. The goal is not only to retain a window overlapping the target, but to preserve enough visual evidence to distinguish the event interior from its immediate pre- and post-event context.

### Contribution 3: Recall-constrained temporal retrieval

The temporal stage will be optimized and calibrated for candidate recall under a duration or encoding budget, rather than Top-$K$ ranking accuracy alone. An uncertainty-aware fallback will widen temporal coverage when retrieval confidence is low. This directly addresses irreversible Stage 1 failure.

### Contribution 4: Grounding-aware spatio-temporal token allocation

Patch retention will use query relevance, motion-compensated change, spatial uniqueness, boundary likelihood, and uncertainty. More importantly, its masks and budgets will be trained against the final grounding distributions through teacher distillation and span loss, rather than treated as standalone heuristics.

### Contribution 5: An evaluation protocol that separates search, representation, and localization

The project will report candidate recall, boundary coverage, localization accuracy, and component-wise cost. This separates gains from temporal search, vision-encoder savings, downstream token compression, and final boundary refinement.

---

## 4. Methodology

### 4.1 System overview

Q-MoG contains five computational stages:

```text
Query-independent coarse index
          +
       Text query
          │
          ▼
Multi-scale recall-oriented temporal retrieval
          │
          ▼
Uncertainty-aware window/frame budget allocation
          │
          ▼
Selective expert encoding and grounding-aware token sparsification
          │
          ▼
Sparse start/end grounding head
          │
          ▼
High-FPS local boundary refinement
```

The index is reusable across queries. The remaining computation is query-conditioned.

### 4.2 Stage 0: query-independent hierarchical index

At a low frame rate $f_c$ and spatial resolution $R_c$, a lightweight encoder $\phi_c$ produces

\[
c_t=\phi_c(f_t), \qquad c_t\in\mathbb R^{d_c}.
\]

The system stores timestamped visual embeddings and, where available, auxiliary features:

\[
x_t=\left[c_t;,a_t^{\mathrm{ASR}};,o_t^{\mathrm{OCR}};,v_t^{\mathrm{codec}}\right].
\]

Scene boundaries and pooled segment embeddings are precomputed at multiple scales. For live video, the same representation can be maintained as a rolling hierarchy with recent fine-grained memory and progressively compressed older memory.

This stage is reported separately because offline indexing changes amortized query cost but not the one-time cost of processing a new video.

### 4.3 Stage 1: multi-scale temporal retrieval

Construct overlapping windows at several durations:

\[
\mathcal W=\bigcup_{\ell=1}^{L}\mathcal W^{(d_\ell)},
\qquad d_\ell\in\{8,16,32,64\}\text{ seconds}
\]

with dataset-specific adjustment for typical event duration. A pooling network produces $z_i$ for each window $W_i$, and a query encoder produces $e_q$. The base relevance score is

\[
s_i^{\mathrm{sem}}=
\cos(P_v z_i,P_q e_q).
\]

When modalities are present, the retriever computes

\[
s_i=
s_i^{\mathrm{sem}}
+\lambda_a s_i^{\mathrm{ASR}}
+\lambda_o s_i^{\mathrm{OCR}}
+\lambda_n s_i^{\mathrm{novelty}}.
\]

Temporal non-maximum suppression or a submodular diversity objective selects candidates without wasting budget on nearly identical overlapping windows.

#### Boundary halos

Each selected window receives a context halo:

\[
\widetilde W_i=
[t_i^{s}-h_i^-,\,t_i^{e}+h_i^+].
\]

The halo sizes may be predicted from query duration priors and retrieval uncertainty rather than fixed globally.

#### Recall-aware selection

Let $o_i=\operatorname{tIoU}(W_i,y^*)$ or a softer overlap target. Retrieval is trained with positive windows that overlap the target and hard negatives from semantically similar but temporally incorrect regions. In addition to ranking loss, Q-MoG uses a differentiable approximation to candidate coverage:

\[
P_{\mathrm{cover}}
=1-\prod_i\left(1-g_i\,o_i\right),
\]

where $g_i\in[0,1]$ is a relaxed selection gate. The recall loss is

\[
\mathcal L_{\mathrm{cover}}=-\log(P_{\mathrm{cover}}+\epsilon).
\]

At inference, calibration on a validation set chooses a threshold or budget policy targeting a specified CandidateRecall@Budget. This is an empirical recall target, not a distribution-free guarantee.

### 4.4 Stage 2: expert-compute allocation

To reduce vision-encoder computation—not merely downstream token count—the allocation decision must precede or occur within expensive encoding. Q-MoG will evaluate two realizations:

1. **Sidekick/expert routing:** use the coarse index to choose frames for a full expert encoder.
2. **Early-exit token routing:** process all selected frames through patch embedding and a small number of shallow ViT blocks, then route only retained tokens through deeper blocks.

For candidate frame $t$, an allocator predicts a frame budget

\[
K_t=K_{\min}+
\operatorname{Round}
\left[(K_{\max}-K_{\min})
\sigma(g_\psi(s_t,b_t,u_t,d_q))\right],
\]

where $s_t$ is frame-query relevance, $b_t$ is boundary likelihood, $u_t$ is model uncertainty, and $d_q$ is a query-complexity representation. The budgets obey a global constraint

\[
\sum_{t\in\mathcal C}K_t\le B_{\mathrm{token}}.
\]

A normalized allocation, knapsack relaxation, or differentiable sorting operator can enforce this constraint during training.

### 4.5 Stage 3: query-conditioned motion and boundary-aware token sparsification

Let $E_{t,p}\in\mathbb R^d$ be the expert or shallow-block representation of patch $p$ in frame $t$.

#### Query relevance

\[
a_{t,p}=\cos(W_vE_{t,p},W_qe_q).
\]

#### Motion-compensated change

Raw same-position feature difference is sensitive to camera motion. Q-MoG instead estimates a correspondence field $F_{t-1\rightarrow t}$ from optical flow, codec motion vectors, or feature matching:

\[
m_{t,p}=1-
\cos\!\left(
E_{t,p},
\operatorname{Warp}(E_{t-1},F_{t-1\rightarrow t})_p
\right).
\]

A lower-cost variant subtracts estimated global camera motion and uses the residual.

#### Static visual uniqueness

\[
u_{t,p}=1-
\max_{p'\in\mathcal N(p)}
\cos(E_{t,p},E_{t,p'}).
\]

This term protects distinctive static evidence that motion-only pruning would remove.

#### Boundary likelihood

A temporal boundary network consumes adjacent frame representations and the query:

\[
b_t=\sigma\left(g_b(E_{t-w:t+w},e_q)\right).
\]

It is supervised using soft labels centered on $t_s^*$ and $t_e^*$. A simple non-learned ablation uses the absolute change in frame-query relevance.

#### Uncertainty

Uncertainty $u_t^{\mathrm{pred}}$ may be estimated using predictive entropy of coarse start/end distributions, retrieval margin, or variance across lightweight stochastic heads. It is used to increase—not decrease—coverage near ambiguous regions.

#### Retention policy

An initial interpretable policy scores tokens as

\[
r_{t,p}=
\alpha a_{t,p}+
\beta m_{t,p}+
\gamma u_{t,p}+
\delta b_t+
\eta u_t^{\mathrm{pred}}.
\]

The full model replaces fixed coefficients with a query-conditioned scoring network. Top-$K_t$ selection is hard at inference and approximated during training with a straight-through or differentiable Top-$K$ estimator.

#### Pruning versus merging

High-scoring tokens are retained individually. Low-motion, mutually similar tokens are clustered and merged:

\[
\bar E_j=
\frac{\sum_{(t,p)\in\mathcal S_j}w_{t,p}E_{t,p}}
{\sum_{(t,p)\in\mathcal S_j}w_{t,p}}.
\]

Each merged token carries temporal-span and spatial-location metadata. A minimum static-context quota prevents the model from allocating the entire budget to motion.

### 4.6 Stage 4: sparse grounding head

The retained sequence is processed by a cross-modal grounding transformer. Tokens are pooled or routed into timestamped frame states $H_t$. Dedicated start and end heads predict

\[
p_s(t)=\operatorname{softmax}(g_s(H_t)),
\qquad
p_e(t)=\operatorname{softmax}(g_e(H_t)).
\]

The interval is decoded with a learned or factorized span score:

\[
(\hat t_s,\hat t_e)=
\arg\max_{t_s\le t_e}
p_s(t_s)p_e(t_e)p_{\mathrm{span}}(t_s,t_e).
\]

This avoids treating free-form timestamp generation by an LLM as the sole localization mechanism. An LLM may still provide semantic representations or auxiliary rationale supervision.

### 4.7 Stage 5: local boundary refinement

Around each coarse boundary, the system extracts a local interval of radius $r$, resamples at $f_r$ FPS or at the original frame rate, and applies a small refiner:

\[
\Delta \hat t_s,\Delta \hat t_e
=g_r(V_{\mathrm{local}},q).
\]

The final predictions are

\[
\hat t_s'=\hat t_s+\Delta\hat t_s,
\qquad
\hat t_e'=\hat t_e+\Delta\hat t_e.
\]

Timestamp precision will be described in terms of sampling resolution and measured boundary error. The work will not claim millisecond accuracy unless supported by the source frame rate and evaluation.

---

## 5. Training strategy

### 5.1 Objective

The full loss is

\[
\mathcal L=
\mathcal L_{\mathrm{span}}
+\lambda_r\mathcal L_{\mathrm{rank}}
+\lambda_v\mathcal L_{\mathrm{cover}}
+\lambda_b\mathcal L_{\mathrm{boundary}}
+\lambda_d\mathcal L_{\mathrm{distill}}
+\lambda_c\mathcal L_{\mathrm{compute}}.
\]

#### Span loss

\[
\mathcal L_{\mathrm{span}}=
-\log p_s(t_s^*)
-\log p_e(t_e^*)
+\lambda_{\mathrm{IoU}}\mathcal L_{\mathrm{IoU}}(\hat y,y^*).
\]

#### Retrieval ranking loss

A contrastive or listwise ranking loss places target-overlapping windows above hard negatives. Positives at multiple temporal scales reduce bias toward one event duration.

#### Boundary loss

Soft boundary targets are Gaussian or triangular distributions around the annotated start and end. The loss includes both boundary classification and boundary-neighborhood token coverage:

\[
\mathcal L_{\mathrm{boundary}}
=\mathcal L_{\mathrm{BCE}}(b,b^*)
+\lambda_{bc}
\sum_{t\in\mathcal B^*}
\max(0,\kappa-\sum_p g_{t,p}),
\]

where $\mathcal B^*$ contains frames near the two ground-truth boundaries and $\kappa$ is the minimum retained evidence quota.

#### Dense-teacher distillation

An unpruned teacher supplies start/end distributions and optionally intermediate cross-modal states:

\[
\mathcal L_{\mathrm{distill}}=
D_{\mathrm{KL}}(P_T^s\Vert P_S^s)
+D_{\mathrm{KL}}(P_T^e\Vert P_S^e)
+\lambda_h\lVert H_T-\Pi(H_S)\rVert_2^2.
\]

This trains selection to preserve the teacher's grounding behavior, not generic feature similarity.

#### Compute loss

The primary budget loss uses component-aware estimated cost:

\[
\mathcal L_{\mathrm{compute}}=
\left|
\frac{\widehat C(A)}{C_{\mathrm{dense}}}-\rho_C
\right|,
\]

rather than token retention alone. A latency lookup table measured on the target hardware can replace FLOPs estimates during later training or policy tuning.

### 5.2 Curriculum

Training will proceed in four phases:

1. Train or adapt the dense grounding teacher.
2. Train the multi-scale retriever for high candidate recall.
3. Train the sparse allocator and grounding head with fixed retrieval candidates and teacher distillation.
4. Jointly fine-tune retrieval, allocation, grounding, and refinement with relaxed gates and progressively tighter compute budgets.

This curriculum reduces instability from simultaneously learning retrieval and discrete sparsification.

### 5.3 Training-free MVP

Before joint training, a reproducible MVP will use frozen encoders, cosine-based multi-scale retrieval, temporal NMS, fixed halos, query-patch cosine, motion-compensated feature change, fixed weighted ranking, token merging, and an existing grounding backbone. This isolates whether the proposed evidence signals are useful before investing in end-to-end optimization.

---

## 6. Research hypotheses

The numerical values below are preregistered experimental targets, not promised outcomes.

**H1 — High-recall search.** Retaining 10–20% of video duration will preserve at least 95% candidate recall on selected long-video VTG benchmarks.

**H2 — Grounding-aware sparsification.** Within retrieved windows, retaining 20–30% of expert visual tokens will cause only a small reduction in $R@1,\mathrm{IoU}=0.5$ relative to processing the same windows densely.

**H3 — Joint efficiency.** The complete system will retain approximately 2–6% of the dense baseline's downstream visual tokens and yield a better accuracy–latency Pareto frontier than temporal-only, token-only, and naive cascaded baselines.

**H4 — Boundary preservation.** At equal compute, explicit boundary-aware selection will reduce start- and end-time MAE and improve $R@1,\mathrm{IoU}=0.7$ compared with generic token pruning.

**H5 — Robustness.** Query-conditioned allocation with a protected static-context budget will outperform motion-only selection on static-evidence queries and motion compensation will improve results on camera-motion-heavy videos.

**H6 — Adaptive allocation.** A learned global allocator will outperform fixed temporal and spatial retention ratios at matched measured latency.

---

## 7. Experimental plan

### 7.1 Datasets

The main evaluation will combine long-form and conventional VTG datasets:

- **Ego4D-NLQ:** long, egocentric, camera-motion-heavy video.
- **MAD:** long movie videos and language queries.
- **Charades-STA:** conventional short-video grounding for comparison with established VTG literature.
- **ActivityNet Captions:** varied event durations and broader activities.
- **ExtremeWhenBench:** hour-scale evaluation designed to separate search from localization.

Dataset licenses, availability, annotation conventions, and official evaluation code will be audited before final selection. If compute is limited, the primary pair will be Ego4D-NLQ and MAD, with Charades-STA used for controlled ablation.

### 7.2 Baselines

The comparison set will include:

1. dense full-video or maximally dense feasible baseline;
2. uniform frame sampling;
3. temporal retrieval only;
4. token pruning only;
5. CONE-style query-guided windows;
6. DeCafNet-style sidekick/expert selection;
7. PruneVid;
8. MoPrune;
9. QTSplus or an equivalent query-aware adaptive token selector;
10. naive temporal retrieval followed by motion pruning;
11. Q-MoG without joint training;
12. full Q-MoG.

All efficiency comparisons will match either accuracy, measured latency, expert-encoder FLOPs, or downstream token count. Token-matched results alone will not be presented as end-to-end efficiency comparisons.

### 7.3 Accuracy and retrieval metrics

- $R@1$ at temporal IoU thresholds $0.3,0.5,0.7$;
- mean temporal IoU;
- start-time MAE and end-time MAE in seconds;
- CandidateRecall@Budget;
- selected-duration coverage;
- boundary-neighborhood recall;
- calibration of retrieval confidence;
- performance stratified by target duration and video duration.

Candidate recall will count a query as covered under multiple overlap definitions: any overlap, containment of both boundaries, and candidate tIoU above a threshold. This prevents an overly permissive metric from hiding boundary truncation.

### 7.4 Efficiency metrics

- percentage of video duration retrieved;
- frames processed by coarse and expert encoders;
- tokens entering shallow and deep vision blocks;
- tokens entering the projector and grounding model;
- component and total FLOPs;
- peak GPU memory;
- preprocessing/indexing time and storage;
- amortized per-query latency;
- cold-start end-to-end latency;
- time to first output, where an MLLM is used;
- boundary-refinement overhead.

All latency results will specify hardware, precision, batch size, caching, video decoding policy, and whether index construction is included.

### 7.5 Required ablations

| Ablation | Question answered |
|---|---|
| Query relevance only | How far semantics alone can go |
| Motion only | Whether dynamics are sufficient |
| Query + motion | Complementarity of semantic and dynamic evidence |
| Query + motion + uniqueness | Protection of distinctive static evidence |
| Add boundary likelihood | VTG-specific value of boundary cues |
| Add uncertainty | Value of conservative allocation under ambiguity |
| Raw feature difference | Low-cost motion baseline |
| Camera-compensated difference | Robustness to egomotion |
| Fixed halos vs adaptive halos | Effect on candidate and boundary recall |
| Fixed vs adaptive frame/token budgets | Value of global allocation |
| Hard pruning vs merging | Compression–context tradeoff |
| Post-encoder vs early/intermediate pruning | Where vision-encoder savings arise |
| Without local refinement | Contribution to fine localization |
| Independent vs joint/distilled training | Whether task-aware optimization matters |
| No boundary-coverage loss | Whether gains come from explicit evidence protection |

### 7.6 Slice analysis

Queries will be divided into:

- dynamic actions;
- static objects or states;
- speech-dependent events;
- OCR-dependent events;
- high versus low camera motion;
- short, medium, and long targets;
- single-event versus repeated-event queries;
- high- versus low-retrieval-confidence cases.

Where annotations are unavailable, a documented automatic classifier followed by manual validation of a sample will be used.

### 7.7 Statistical protocol

Primary results will use the official test splits. Ablations will report at least three random seeds when training variance is material. Query-level paired bootstrap confidence intervals will be computed for differences in mIoU, boundary MAE, and recall. Pareto dominance will be evaluated over multiple budgets rather than at one handpicked retention rate. Hyperparameters and thresholds will be selected only on validation data.

---

## 8. Success criteria

The project will be considered successful if it demonstrates all of the following:

1. **Search viability:** at least 95% candidate recall at a practically meaningful temporal budget on one long-video benchmark.
2. **Boundary benefit:** statistically supported improvement in boundary MAE or high-IoU recall over generic pruning at matched cost.
3. **Joint-allocation benefit:** a superior accuracy–latency or accuracy–FLOPs Pareto frontier over the naive temporal-plus-token cascade.
4. **Actual encoder savings:** at least one implementation reduces expert vision-encoder work through pre-encoder frame routing or intermediate ViT pruning.
5. **Robustness evidence:** reduced degradation on static-evidence and camera-motion-heavy subsets.

The strongest paper outcome would validate all five. If only downstream token savings are achieved, the claims will be limited to projector/grounder efficiency rather than end-to-end video efficiency.

---

## 9. Risks and mitigation

### Retrieval errors dominate final performance

**Risk:** the correct event is discarded before grounding.

**Mitigation:** optimize candidate coverage directly; retain diverse candidates; calibrate uncertainty; increase the budget when confidence is low; use temporal halos.

### Static evidence is removed

**Risk:** motion-based selection deletes query-critical signs, objects, subtitles, or terminal states.

**Mitigation:** include query relevance and static uniqueness; reserve a static-context quota; merge rather than delete redundant background.

### Camera motion overwhelms object motion

**Risk:** egomotion makes most tokens appear salient.

**Mitigation:** compare codec vectors, global-motion compensation, optical flow, and feature correspondence; use scene-level normalization.

### Boundary labels are noisy

**Risk:** human start/end annotations are subjective and overly sharp supervision encourages overfitting.

**Mitigation:** use soft boundary targets, tolerance-aware metrics, and local contrast between event and surrounding context.

### Discrete selection destabilizes joint training

**Risk:** gradients through Top-$K$ gates are biased or unstable.

**Mitigation:** staged training, dense-teacher distillation, soft gates followed by hardening, and comparison of straight-through and differentiable sorting estimators.

### Added modules erase efficiency gains

**Risk:** flow estimation, uncertainty heads, or refinement cost more than they save.

**Mitigation:** prioritize codec motion vectors or low-resolution feature matching; report every component; retain a low-cost variant; use measured latency in allocation.

### Novelty is narrowed by concurrent work

**Risk:** rapidly developing long-video tokenization research introduces overlapping techniques.

**Mitigation:** anchor the contribution in the complete grounding-specific formulation and evidence: explicit boundary coverage, recall-constrained search, joint task-aware allocation, and decomposition of end-to-end cost. Repeat the literature search before submission and avoid absolute priority claims.

---

## 10. Implementation plan and milestones

### Phase 1 — Reproduction and measurement (Weeks 1–4)

- Select one grounding backbone and two primary datasets.
- Reproduce a dense baseline and a query-guided temporal-retrieval baseline.
- Build a component-wise latency/FLOPs profiler.
- Implement CandidateRecall@Budget and boundary-coverage evaluation.

**Deliverable:** reliable dense and temporal-only baselines with a reproducible efficiency report.

### Phase 2 — Training-free Q-MoG MVP (Weeks 5–8)

- Build the reusable multi-scale index.
- Add diverse Top-$K$, fixed halos, and uncertainty from retrieval margins.
- Implement query relevance, raw and compensated motion, static uniqueness, fixed adaptive budgets, and token merging.
- Run the core signal and budget ablations.

**Go/no-go test:** boundary-aware scoring must improve high-IoU recall or boundary MAE over naive motion pruning at matched retained tokens.

### Phase 3 — Grounding-aware learning (Weeks 9–14)

- Train the boundary head and sparse grounding head.
- Add dense-teacher distillation and boundary-coverage loss.
- Train adaptive frame and token allocation.
- Compare independent and joint training.

**Deliverable:** full accuracy–compute Pareto curves on the development datasets.

### Phase 4 — Encoder savings and refinement (Weeks 15–18)

- Implement sidekick/expert routing or intermediate ViT pruning.
- Add high-FPS local boundary refinement.
- Measure end-to-end latency with and without cached indexing.

**Deliverable:** verified vision-encoder, downstream-token, memory, and latency savings.

### Phase 5 — Generalization and paper study (Weeks 19–24)

- Evaluate on additional datasets.
- Complete query-type, camera-motion, and target-duration analyses.
- Run statistical tests, limitations analysis, and final novelty audit.
- Release code, configurations, index metadata, and evaluation scripts where licenses permit.

---

## 11. Expected outcomes

The expected scientific outcome is a clearer account of where efficiency can safely be introduced in long-video grounding. The project should establish whether:

1. high-recall search can reduce temporal coverage without truncating boundaries;
2. token budgets should depend on boundary likelihood and uncertainty rather than relevance alone;
3. static context and motion require explicit competing quotas;
4. joint training preserves grounding behavior better than independently designed pruning stages; and
5. downstream token reduction translates into actual encoder and end-to-end latency improvements.

Even a negative result would be informative if it identifies the budget at which candidate recall collapses or shows that boundary refinement, rather than spatial pruning, dominates high-IoU performance.

---

## 12. Limitations and responsible research considerations

Q-MoG relies on coarse representations that may underperform for small objects, rare actions, non-visual evidence, or culturally specific concepts. ASR and OCR can introduce language and accent biases. Aggressive pruning reduces auditability because discarded evidence is unavailable to the final model. The system should therefore log selected windows and masks, expose retrieval confidence, and support a conservative mode that expands coverage for uncertain queries.

Long-video datasets may contain personally identifiable or copyrighted material. Experiments must follow dataset licenses, minimize unnecessary storage of decoded frames, and avoid releasing extracted content when redistribution is prohibited. Efficiency improvements should be reported alongside their preprocessing and hardware costs rather than framed as universal energy reductions.

---

## 13. Proposed paper positioning

### Working title

> **Q-MoG: Query-Conditioned Motion and Boundary-Aware Pruning for Efficient Long-Video Temporal Grounding**

### One-sentence positioning

> Existing long-video methods primarily optimize temporal search or generic visual-token efficiency; Q-MoG studies their joint design for temporal grounding, where candidate recall, static query-relevant evidence, and event boundaries must be preserved under a global compute budget.

### Claims to avoid

- “The first method to combine temporal filtering and token pruning.”
- “Motion is sufficient to identify useful grounding tokens.”
- “A 98% token reduction implies a 98% end-to-end speedup.”
- “Millisecond-accurate grounding” without matching temporal evidence and evaluation.
- Fixed claims of 80–90% temporal removal or 75–80% token removal before experiments.

### Defensible claim template

> At matched measured compute, Q-MoG improves the localization–efficiency Pareto frontier by jointly allocating temporal and spatial evidence budgets with explicit candidate-coverage and boundary-preservation objectives.

This claim becomes publishable only if the comparisons show gains over both specialized components and their strongest naive composition.

---

## References

1. Hou et al. [CONE: An Efficient COarse-to-fiNE Alignment Framework for Long Video Temporal Grounding](https://arxiv.org/abs/2209.10918). ACL 2023.
2. Ye et al. [T*: Re-thinking Temporal Search for Long-Form Video Understanding](https://arxiv.org/abs/2504.02259). CVPR 2025.
3. Huang, Zhou, and Han. [PruneVid: Visual Token Pruning for Efficient Video Large Language Models](https://aclanthology.org/2025.findings-acl.1024/). Findings of ACL 2025.
4. [DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](https://arxiv.org/abs/2505.16376). 2025.
5. [Natural-Language Temporal Grounding in Hour-Long Videos is a Search Problem: A Benchmark and Empirical Decomposition](https://arxiv.org/abs/2606.12300). 2026.
6. [Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models](https://arxiv.org/abs/2511.11910). 2025.
7. Hong et al. [MoPrune: Scene-Guided Motion-Aware Token Pruning for Efficient Video Large Language Models](https://aclanthology.org/2026.findings-acl.344/). Findings of ACL 2026.
8. Tang et al. [Q-Fold: Query-Aware Focus-Context Spatio-Temporal Folding for Long Video Understanding](https://arxiv.org/abs/2606.12125). 2026.
9. Pan et al. [TimeSearch: Hierarchical Video Search with Spotlight and Reflection for Human-like Long Video Understanding](https://arxiv.org/abs/2504.01407). 2025.
