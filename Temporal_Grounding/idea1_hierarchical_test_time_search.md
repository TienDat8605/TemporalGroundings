# Idea 1: Hierarchical Test-Time Search for Generalist Temporal Grounding

**Status:** v1 implemented in `TimeLens2/evaluation/run_omtg_search.py`  
**Depends on:** TimeLens2 and related VTG-MLLM literature  
**Primary target model:** frozen TimeLens2 (or any strong open temporal grounder) as a black-box localizer  

---

## 1. One-sentence claim

Under a **fixed inference compute budget**, multi-pass **hierarchical frame allocation** (coarse candidate discovery → local dense re-grounding → set merge) improves long-video and multi-span temporal grounding **without** fine-tuning, by closing the train/data vs inference mismatch in TimeLens2-style systems.

### Revised experimental goal (2026-07-23)

The first question is now deliberately narrow:

> On the official 320-query **OMTG Bench**, does embedding-routed local
> TimeLens2-4B inference improve exact multi-span grounding over three matched
> scheduling controls, with frozen weights on one Colab T4?

This replaces the earlier VUE/QVHighlights-first plan. OMTG has 2--20 ground
truth spans per query and evaluates set cardinality directly; it therefore
tests the pain point that motivated TimeLens2-93K. `TimeLens2-93K` remains the
training source, not an evaluation benchmark. QVHighlights is only an optional
regression check after the OMTG result.

The v1 policy is fixed before running the benchmark:

1. deterministic content-aware windows constrained to 20--60 seconds, with a
   two-second boundary overlap and 45-second/four-second-overlap uniform fallback;
2. four sparse frames per window and the query encoded by
   `Qwen/Qwen3-VL-Embedding-2B`;
3. cosine ranking with
   `K=min(8,max(2,ceil(sqrt(number_of_windows))))`;
4. all retained windows queried independently with frozen
   `MCG-NJU/TimeLens2-4B`, permitting an empty set or every clip-relative span;
5. global timestamp mapping, high-tIoU duplicate suppression, and gap merge at
   one second.

The four schedules are **uniform-one-shot**, **full-video-multipass**,
**uniform-window-local**, and **embedding-window-local** at nominal 32/64
decoded-frame tiers. Router frames are charged to the proposed method. Every
record stores frames, model calls, synchronized single-T4 model time, wall
time, and peak VRAM. A result is called compute-matched only within 5% aggregate
GPU time; otherwise it is reported only on the accuracy--compute Pareto curve.
Primary metric is OMTG **EtF1**; secondary metrics are C-Acc, tF1@0.3/0.5/0.7,
union tIoU, cardinality error, router recall, and offline oracle router recall.

This use of Qwen3-VL-Embedding is a new test-time router, not a reproduction of
the TimeLens2-93K annotation ensemble. The original pipeline used large Qwen/Kimi
caption and proposal teachers, dual temporal localizers, and only then a
Qwen3-VL-Embedding semantic verification stage.

---

## 2. Motivation

### 2.1 What TimeLens2 fixed

Zhu et al. (2026) argue that generalist video temporal grounding fails for two structural reasons:

1. **Supervision mismatch:** long-video labels from brittle one-pass annotation miss repeats, confuse similar occurrences, and blur boundaries.
2. **Optimization mismatch:** next-token SFT does not optimize interval geometry; pure tIoU rewards plateau at zero for all non-overlapping predictions; multi-span matching rewards are fragile.

Their fixes are train-time:

- **TimeLens2-93K:** staged label construction (hierarchical captions → coarse proposals → dual-agent local grounding → consensus + semantic check → local boundary refine).
- **SFT → GRPO** with reward  
  \(R = R_{\mathrm{tIoU}} + R_{\mathrm{TW}} - \mathbf{1}_{\mathrm{invalid}}\),  
  where \(R_{\mathrm{TW}}\) is match-free 1D Wasserstein geometry on merged supports.

This is strong work on **what to train** and **how to score sets during RL**.

### 2.2 What stays broken at inference

Deployed inference in TimeLens2 (and most VTG-MLLMs) is still approximately:

```text
(v, q) → uniform frame sample (e.g. 2 fps, ≤512 frames, ~16K visual tokens)
       → single forward decode
       → parse interval set Ŷ
```

For long videos, a global frame cap forces **uniform temporal downsampling**. Brief, late, or repeated evidence can disappear before the language model reasons (see also momentary-event failures under sparse sampling; Liu et al., 2026, *Moment-Video*).

**Structural irony:** TimeLens2-93K *constructs* labels with multi-scale search and local densification, but the *model* does not *use* multi-scale search at test time. The paper’s main axis is data + RL reward; **adaptive / hierarchical test-time frame allocation is not first-class** (limitations focus on teacher quality and future spatiotemporal extension; Zhu et al., 2026, App. A.6).

### 2.3 Research niche

Recent “zoom / tree / dynamic granularity” methods improve long-video understanding, but usually via **additional SFT+RL** (tool policies, navigation policies). A low-resource angle:

> Treat a strong open grounder (e.g. TimeLens2-2B/4B) as frozen.  
> Redesign only **where frames and passes are spent** under a matched budget.

---

## 3. Problem statement

**Task.** Generalist temporal grounding (Zhu et al., 2026):

\[
(v, q) \mapsto \mathcal{Y} = \{[s_k, e_k]\}_{k=1}^{K},
\]

variable cardinality \(K\), long or short \(v\), declarative or question-form \(q\).

**Baseline inference \(f_0\).** Single-pass model with global frame budget \(B\):

\[
\hat{\mathcal{Y}}_0 = f_0(v, q; \mathrm{SampleUniform}(v, B)).
\]

**Proposed inference \(f_H\).** Hierarchical procedure with the **same** total frame budget \(B\) (or same FLOPs / wall-clock class):

\[
\hat{\mathcal{Y}}_H = f_H(v, q; B),
\]

where \(f_H\) allocates frames across stages (global recall → local precision → merge), calling the **same frozen** grounder \(f\) multiple times on different frame sets.

**Success.** Higher mIoU / R@IoU on long-video and multi-span regimes at equal budget, or a better accuracy–compute Pareto curve.

---

## 4. Method

### 4.1 Design principle

Mirror the **logical cascade** of TimeLens2-93K (Zhu et al., 2026, §3.1), but at **test time** with one frozen localizer:

| Data pipeline stage | Test-time analogue |
|---------------------|--------------------|
| Hierarchical captions / global context | Cheap global or chunked scan |
| Coarse proposal set \(\mathcal{P}(q)\) | Candidate windows \(W_{(1)},\ldots,W_{(K)}\) |
| Dual-agent local grounding on short clips | Dense re-grounding on candidates |
| Consensus + semantic verification | Optional multi-sample / score filter |
| ±3s boundary refinement | Optional local boundary densify |
| Merge near-adjacent spans | Global temporal NMS / gap-merge |

No weight updates required for the core method.

### 4.2 Budget model

Let \(B\) = total number of **decoded frames** processed by the vision tower across all calls (primary accounting unit). Variants may account forward passes or TFLOPs if double-encoding matters.

Split, for example:

\[
B = B_{\mathrm{global}} + B_{\mathrm{local}} + B_{\mathrm{refine}},
\]

with defaults such as \(B_{\mathrm{global}}=\lfloor B/2 \rfloor\), remainder on local (+ optional refine).  
**Critical experimental rule:** every claim vs one-shot TimeLens2 must state the budget policy.

### 4.3 Stage 1 — Global candidate discovery (recall-first)

**Goal:** find windows that *might* contain evidence; prefer recall over precision.

**Windowing.**

1. **Scene-based:** content-aware cuts (e.g. PySceneDetect-style), clip length constrained to ~20–60s (same spirit as TimeLens2 caption clips; Zhu et al., 2026, §3.1).  
2. **Uniform fallback:** fixed length \(L\) with hop \(h < L\) (**overlap required** so boundary events are not lost).

Denote windows \(\{W_i = [a_i, b_i]\}\).

**Scoring each window** (pick one or combine):

| Option | Procedure | Cost | Notes |
|--------|-----------|------|-------|
| A. Frozen grounder, sparse | Few frames in \(W_i\); run \(f\); non-empty \(\hat{\mathcal{Y}}\) or longer support → higher score | Medium | Uses task model itself |
| B. Embedding gate | Text–video similarity (e.g. Qwen3-VL-Embedding; Li et al., 2026) on sparsely sampled frames | Low | Same family already used inside TimeLens2-93K semantic check |
| C. Hybrid | Embedding shortlist → sparse \(f\) confirm | Low–med | Good default |

Keep top-\(K\) windows, or all above threshold \(\tau\).  
For multi-span queries, allow **several disjoint** survivors (do not force a single mode).

**Output of stage 1:** candidate set \(\mathcal{C} = \{W_{(1)},\ldots,W_{(K)}\}\), optionally with scores \(s_i\).

### 4.4 Stage 2 — Local dense re-grounding (precision-first)

**Goal:** spend remaining frame budget where stage 1 said evidence lives.

For each \(W_{(i)} \in \mathcal{C}\):

1. Allocate frames \(b_i\) with \(\sum_i b_i \le B_{\mathrm{local}}\), e.g.  
   - equal split \(b_i = B_{\mathrm{local}}/K\), or  
   - proportional to \(s_i\) or to \(|W_{(i)}|\).
2. Sample **denser** frames inside \(W_{(i)}\) (higher effective fps than stage 1 / than global uniform over full \(T\)).
3. Run frozen grounder \(f\) on \((W_{(i)}, q)\) (clip-relative prompts; map times back to global timeline carefully).
4. Collect predicted sets \(\hat{\mathcal{Y}}^{(i)}\) in **global** time.

**Timestamp consistency.** When the model sees different sampling grids on full video vs local clip, map predictions via absolute seconds (or a TimeAnchor-style rule; Chen et al., 2026, *OmniReasoner*), never raw frame indices from mixed fps.

### 4.5 Stage 2b — Optional boundary polish

Mirrors TimeLens2-93K boundary refinement (Zhu et al., 2026, §3.1):

For each predicted boundary \(t \in \{s_k, e_k\}\):

- take neighborhood \([t-\Delta, t+\Delta]\) (e.g. \(\Delta \in [3,5]\) seconds);
- sample at high fps under small \(B_{\mathrm{refine}}\);
- re-query \(f\) to snap that boundary (or re-ground the short span).

Use only if budget remains; ablate on/off.

### 4.6 Stage 3 — Global set merge

Aggregate \(\{\hat{\mathcal{Y}}^{(i)}\}\) into one set \(\hat{\mathcal{Y}}_H\):

1. **Map** all intervals to global time.  
2. **Gap-merge:** merge intervals separated by \(\le \tau_{\mathrm{gap}}\) (TimeLens2 uses ~1s for annotation jitter; Zhu et al., 2026).  
3. **Duplicate suppress:** if two intervals have high pairwise tIoU, keep higher-score or longer.  
4. **Conflict policy:**  
   - multi-span friendly default = **union** of high-scoring disjoint intervals;  
   - if two heavily overlapping disagreements, keep higher stage-2 score or run a tiny local tie-break.

Optional: drop windows whose local output is empty or invalid parse (\(\mathbf{1}_{\mathrm{invalid}}\) spirit from their RL reward).

### 4.7 Algorithm (pseudocode)

```text
Input: video v, query q, frozen grounder f, total frame budget B
Output: interval set Y_hat

# --- budget ---
B_g, B_l, B_r = SplitBudget(B)   # e.g. (B/2, B/2 - B_r, B_r)

# --- stage 1: recall ---
windows = BuildWindows(v)        # scene cuts or fixed L, hop h < L
scores = []
for W in windows:
    frames = Sample(W, n=n_sparse)   # counts toward B_g
    y, sc = ScoreWindow(f or embed, frames, q)
    scores.append((W, y, sc))
C = TopK(scores, K) or Threshold(scores, tau)

# --- stage 2: precision ---
alloc = Allocate(B_l, C)         # equal or proportional
locals = []
for (W, alloc_b) in zip(C, alloc):
    frames = Sample(W, budget=alloc_b)  # denser
    y_local = f(frames, q)              # parse intervals
    y_global = MapToGlobalTime(y_local, W)
    locals.append(y_global)

# --- stage 2b: optional refine ---
if B_r > 0:
    locals = BoundaryPolish(f, v, q, locals, B_r, Delta)

# --- stage 3: merge ---
Y_hat = MergeSets(locals, gap=tau_gap)
return Y_hat
```

### 4.8 Matched baselines (required for honesty)

| Baseline | Description |
|----------|-------------|
| One-shot uniform | Official-style: \(B\) frames over full \(T\), one call |
| One-shot denser (if fits) | More than \(B\) if memory allows — upper reference only |
| Chunked independent | Stage-1 windows each get equal frames, **no** top-K densify redistribution |
| Full-video multi-pass | Multiple full-video samples with same total \(B\) but **no** spatial (temporal) focus — controls “just run more times” |
| Hierarchical (ours) | Stages 1–3 as above |

If hierarchical wins only by using more total frames, the paper fails. If it wins at **equal \(B\)**, the claim holds.

---

## 5. Why this might work

1. **Information geometry of long video:** evidence is sparse; uniform sampling wastes tokens on irrelevant bulk (Motivating diagnosis in long VTG and Moment-Video-style momentary failures).  
2. **Alignment with how labels were made:** TimeLens2 already showed local decisions on short clips are more reliable than one global pass for annotation.  
3. **Frozen model competence:** after SFT+GRPO, TimeLens2 is already a strong *local* grounder; hierarchical search supplies the missing *global search schedule*.  
4. **Multi-span:** top-\(K\) disjoint windows naturally support repeated evidence without fragile one-to-one matching at the search level.

---

## 6. Related work (positioning)

### 6.1 Train-time generalist grounding (starting point, not competitor)

- **TimeLens2** (Zhu et al., 2026): data verification cascade + temporal Wasserstein GRPO; inference remains single-pass uniform sampling.  
- **TimeLens** (prior MCG-NJU line; cited in Zhu et al., 2026): data quality and training recipes for temporal MLLMs.  
- **TimeChat / VTimeLLM / TimeSuite / Grounded-VideoLLM** (Ren et al., 2024; Huang et al., 2024; Zeng et al., 2024; …): temporal instruction tuning and timestamp interfaces — still largely one-shot at test.  
- **RL for VTG** (e.g. VideoChat-R1-style, MUSEG NGIoU matching; cited in Zhu et al., 2026, §2): improve rewards/optimization, not hierarchical test-time frame schedules.

**Gap we attack:** train-time set geometry and labels without test-time multi-scale search.

### 6.2 Coarse-to-fine and zoom at test time (closest neighbors)

- **ScanFocus** (Chen et al., 2026, ECCV): coarse spatio-temporal scan → dense sampling near boundaries for STVG; specialized trained pipeline.  
  *We share coarse→fine structure; differ by frozen MLLM grounder, interval-set outputs, and matched-budget framing for generalist VTG.*

- **OmniReasoner** (Chen et al., 2026): low-cost global preview → learned **zoom-in tool** on temporal intervals (SFT+RL); TimeAnchor for cross-grid time consistency.  
  *We adopt preview→zoom narrative and careful time mapping; avoid tool-policy training by heuristic/frozen search.*

- **VideoTreeSearch (VTS)** (Zhang et al., 2026): grounded long-video QA as search on an adaptive temporal tree (zoom_in / zoom_out / shift / answer) with backtracking; trained navigation.  
  *Richer action space; our v1 is a shallow 2-stage tree without learned policy.*

- **MoD-VLLM** (Feng et al., 2026): modular dynamic-granularity encoding of positive/negative segments with reflection; RL for granularity.  
  *Same “spend tokens on relevant segments” spirit; we externalize allocation as inference search.*

- **Reflect-R1** (Chen et al., 2026): intuition → evidence verification → arbitration for long video.  
  *We verify by denser re-grounding, not only reflective text.*

### 6.3 Diagnoses of sparse sampling

- **Moment-Video** (Liu et al., 2026): momentary answer-critical events fail under sparse sampling / compression; denser FPS is incomplete fix.  
  *Supports our claim that allocation, not only average fps, matters.*

### 6.4 Classical temporal grounding

- Proposal–rank–refine and hierarchical moment retrieval (e.g. TALL / Charades-STA line: Gao et al., 2017; ActivityNet Captions: Krishna et al., 2017; later proposal-free and multi-scale localizers).  
  *We revive classical multi-scale search as a **wrapper around generative MLLMs**, with modern set-valued outputs and compute-Pareto evaluation.*

### 6.5 Test-time compute in LLMs

- Best-of-N, self-consistency, budgeted tree search for text (Wang et al., 2023, self-consistency; Snell et al., 2024, test-time scaling — cite as appropriate in final paper).  
  *Usually vote on tokens/answers. Our core Idea 1 spends budget on **frames and regions**; optional fusion of multiple interval sets (Idea 2) is complementary.*

### 6.6 Positioning sentence

> Prior generalist VTG-MLLMs improve **supervision and RL rewards** (Zhu et al., 2026). Prior long-video agents improve **learned zoom/navigation** (Chen et al., 2026; Zhang et al., 2026). We study **training-free hierarchical frame allocation** that reuses a frozen generalist grounder under **matched inference budgets**.

---

## 7. Experimental protocol

### 7.1 Models

- Local grounder: frozen **MCG-NJU/TimeLens2-4B**.  
- Coarse router: frozen **Qwen/Qwen3-VL-Embedding-2B**.  
- Hardware: one Colab T4; the two models are loaded in separate phases and are never resident together.

### 7.2 Benchmarks (priority order)

| Priority | Benchmark | Why |
|----------|-----------|-----|
| P0 | OMTG Bench (320 queries / 287 videos) | Official 2--20-span GT and set-cardinality metric |
| P1 | QVHighlights-TimeLens | Optional regression/transfer after the OMTG conclusion |
| P2 | VUE-TR-V2 | Optional long-video follow-up only if accessible reproducibly |

Primary metric: **EtF1**. Secondary: C-Acc, tF1@0.3/0.5/0.7,
union tIoU, cardinality error, router recall, offline oracle router recall, and
**frames / calls / synchronized GPU-call time / wall-clock / peak VRAM**.

### 7.3 Main plots

1. **Pareto:** EtF1 vs aggregate GPU time for nominal 32/64-frame tiers.  
2. **Matched table:** compare only rows within 5% aggregate GPU time.  
3. **Schedule table:** one-shot, full-video multipass, uniform-window local, embedding-window local.  
4. **Failure analysis:** embedding router recall vs offline oracle router recall.

### 7.4 Ablations (minimum set)

| ID | Variant | Tests |
|----|---------|--------|
| A0 | One-shot uniform | Baseline |
| A1 | Full-video multipass | “More calls” without focus |
| A2 | Uniform-window local | Local calls without semantic routing |
| A3 | Embedding-window local | Full proposed schedule |
| A4 | Offline oracle router | Diagnose whether the router is the bottleneck |

### 7.5 Success criteria

**Strong result:** higher OMTG EtF1 at matched aggregate GPU time and A1/A2 < A3.  
**Acceptable result:** a better EtF1--compute Pareto point without a matched point.  
**Negative but publishable:** gains vanish at equal \(B\) but oracle stage-1 helps → motivates learned router (future work), still diagnoses bottleneck.

---

## 8. Risks and mitigations

| Risk | Why | Mitigation |
|------|-----|------------|
| Stage-1 miss is fatal | True evidence never densified | Overlap; recall-first threshold; larger \(K\); embedding+model hybrid |
| Latency / cost narrative weak | Reviewers say “just slower” | Matched \(B\) and Pareto; not only max accuracy |
| Timestamp bugs | Clip-relative vs global; mixed fps | Absolute seconds; unit tests; TimeAnchor-style discipline |
| Short-video regression | Extra chunking noise | Disable hierarchy when \(T\) small or \(\lfloor T \cdot \mathrm{fps}\rfloor \le B\) |
| Double-counting multi-span | Overlapping windows | Gap-merge + temporal NMS |
| Prompt/format brittleness | Multiple calls, different crops | Fixed prompt template; invalid parse retry once |

---

## 9. Scope: what this is / is not

**Is**

- Inference-time hierarchical / adaptive frame sampling and multi-pass search.  
- Training-free wrapper around a strong VTG-MLLM.  
- Direct response to TimeLens2’s unused multi-scale data philosophy at test time.

**Is not (unless extended later)**

- New SFT/GRPO recipe or new reward.  
- Learned zoom tool (OmniReasoner) or full temporal tree RL (VTS).  
- Spatiotemporal tubes (paper’s future work).  
- Idea 2 (test-time TW consensus over many decodes) — complementary module, not required for v1.

---

## 10. Optional extensions (after v1 works)

1. **Idea 2 on local stage only:** \(G\) samples per candidate window; fuse with set-IoU / \(R_{\mathrm{TW}}\) self-consistency (Zhu et al., 2026 reward geometry reused at test).  
2. **Tiny learned router:** lightweight classifier on cheap features chooses \(K\) and budget split (small train, not full grounder RL).  
3. **Early exit:** if stage-1 confidence low and \(T\) huge, allocate more \(B_{\mathrm{global}}\); if one window dominates, skip others.  
4. **Streaming:** causal windows for online video.

---

## 11. Writing hooks (paper / proposal)

**Title directions**

- *Search, Then Densify: Hierarchical Test-Time Frame Allocation for Generalist Temporal Grounding*  
- *The Label Pipeline Is the Inference Algorithm: Training-Free Multi-Scale Grounding with Frozen MLLMs*

**Abstract seed**

> Generalist temporal grounding MLLMs such as TimeLens2 improve supervision and reinforcement-learning rewards, yet still localize with a single forward pass under uniform frame sampling. We show this inference schedule is misaligned with both the sparse structure of long-video evidence and the multi-stage search used to build TimeLens2’s own training labels. We propose a training-free hierarchical test-time procedure that splits a fixed frame budget between coarse candidate discovery and local dense re-grounding with a frozen grounder, then merges variable-cardinality interval sets on the global timeline. Under matched compute, hierarchical allocation improves long-form grounding and yields favorable accuracy–efficiency trade-offs without additional fine-tuning.

**Contribution bullets**

1. Diagnose train/data vs inference mismatch in TimeLens2-style VTG-MLLMs.  
2. Propose matched-budget hierarchical test-time search with frozen grounder.  
3. Empirically validate on long-video benches with ablations isolating densification vs mere multi-pass.

---

## 12. Key references (working list)

> Complete bib entries when drafting the paper; arXiv IDs below are anchors from the reading trail.

1. Zhu et al. (2026). *TimeLens2: Generalist Video Temporal Grounding with Multimodal LLMs.* arXiv:2607.17423.  
2. Liu et al. (2026). *Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events.* arXiv:2606.02522.  
3. Chen et al. (2026). *OmniReasoner: Thinking with Long Audio-Video via Native Tool Use.* arXiv:2607.19339.  
4. Chen et al. (2026). *ScanFocus: A Coarse-to-Fine Framework for Spatio-Temporal Video Grounding.* arXiv:2607.13421 (ECCV 2026).  
5. Zhang et al. (2026). *Searching Videos as Trees: Self-Correcting Agents for Grounded Long Video QA.* arXiv:2607.16189.  
6. Feng et al. (2026). *Modularized Dynamic-Granularity Video LLM for Multi-Event Long Video Understanding.* arXiv:2607.15778.  
7. Chen et al. (2026). *Reflect-R1: Evidence-Driven Reflection for Self-Correction in Long Video Understanding.* arXiv:2606.27922.  
8. Gao et al. (2017). TALL / Charades-STA — classical temporal language grounding.  
9. Krishna et al. (2017). ActivityNet Captions.  
10. Grauman et al. (2022). Ego4D (NLQ) — egocentric grounding transfer.  
11. Li et al. (2026). Qwen3-VL-Embedding / reranker — cheap stage-1 gate (as used inside TimeLens2-93K).  
12. Wang et al. (2023). Self-consistency for LLM decoding (test-time compute ancestor).  
13. Related VTG-MLLM / RL lines cited by TimeLens2: TimeChat, VTimeLLM, TimeSuite, MUSEG, VideoChat-R1-style rewards (see Zhu et al., 2026, §2).

---

## 13. Immediate next actions

1. Run the deterministic first-25-query OMTG smoke test on Colab T4.  
2. Inspect parse failures, budget overflow, router recall, and T4 peak memory.  
3. Freeze any necessary bug fix, use a new run name, and run all 320 queries.  
4. Report matched rows only within 5% GPU time; otherwise report the Pareto frontier.  
5. Add QVHighlights regression only after the primary OMTG result is complete.

---

## 14. Bottom line

**Idea 1 = adaptive / hierarchical frame sampling + multi-pass search at inference**, using a frozen generalist grounder, motivated by TimeLens2’s own label cascade and by coarse-to-fine / zoom literature—but **without** requiring tool-policy RL. The scientific core is not “call the model twice”; it is **matched-budget reallocation of frames toward sparse evidence**.
