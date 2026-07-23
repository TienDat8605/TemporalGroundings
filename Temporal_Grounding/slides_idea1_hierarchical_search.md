---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: "Helvetica Neue", "Segoe UI", "Liberation Sans", sans-serif;
    font-size: 26px;
    color: #1a1a1a;
    padding: 44px 52px 56px 52px;
  }
  h1 {
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
    border-bottom: none;
    margin: 0 0 0.5em 0;
  }
  h2 {
    font-size: 20px;
    font-weight: 500;
    color: #555;
    margin: 0 0 0.75em 0;
  }
  p, li { line-height: 1.4; }
  table {
    font-size: 20px;
    width: 100%;
    border-collapse: collapse;
  }
  th {
    font-weight: 600;
    text-align: left;
    border-bottom: 1.5px solid #222;
    padding: 0.28em 0.4em;
  }
  td {
    border-bottom: 1px solid #e6e6e6;
    padding: 0.28em 0.4em;
    vertical-align: top;
  }
  code, pre {
    font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
    font-size: 16px;
    background: #f5f5f5;
  }
  pre {
    padding: 0.8em 1em;
    border-radius: 5px;
    border: 1px solid #e6e6e6;
    line-height: 1.45;
    margin: 0.5em 0;
  }
  footer { font-size: 13px; color: #888; }
  header { font-size: 12px; color: #999; text-align: right; }
  section.center {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .cite {
    font-size: 15px;
    color: #666;
    margin-top: 0.75em;
    line-height: 1.35;
  }
  .quiet { color: #666; font-size: 18px; }
  .formula { font-size: 24px; text-align: center; margin: 0.9em 0; }
  .label {
    font-size: 13px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #777;
    margin-bottom: 0.35em;
  }
---

<!--
Idea 1 deck · hierarchical test-time search
Source: Temporal_Grounding/idea1_hierarchical_test_time_search.md
-->

<!-- _class: center -->
<!-- _footer: "" -->
<!-- _paginate: false -->

# Hierarchical test-time search
## for generalist temporal grounding

<span class="quiet">Idea 1 · training-free · matched inference budget</span>

<br>

<span class="cite">
Depends on TimeLens2 [Zhu et al., 2026, arXiv:2607.17423]<br>
v1 target: frozen TimeLens2-4B + Qwen3-VL-Embedding-2B router
</span>

---

<!-- _header: Idea 1 · Claim -->
<!-- _footer: idea1_hierarchical_test_time_search.md §1 -->

# One-sentence claim

Under a **fixed inference budget**, multi-pass  
**hierarchical frame allocation**

```text
coarse candidate discovery
        → local dense re-grounding
        → set merge
```

improves long-video / multi-span grounding  
**without fine-tuning**.

<span class="cite">Closes train/data vs inference mismatch in TimeLens2-style systems [Zhu et al., 2026].</span>

---

<!-- _header: Idea 1 · Motivation -->
<!-- _footer: Zhu et al., 2026, §1–3; App. A.6 -->

# What TimeLens2 already fixed

| Train-time fix | Role |
|----------------|------|
| TimeLens2-93K cascade | verified multi-span labels |
| SFT → GRPO + tIoU + $R_{\mathrm{TW}}$ | set geometry in RL |

```text
labels built by:  propose → local verify → refine
model deployed as:  one pass · uniform frames
```

**Irony:** multi-scale search makes the **data**;  
inference does **not** multi-scale search.

<span class="cite">[Zhu et al., 2026]. Adaptive test-time frame allocation not first-class (cf. App. A.6).</span>

---

<!-- _header: Idea 1 · Motivation -->
<!-- _footer: Zhu et al., 2026; Liu et al., 2026 -->

# What stays broken at inference

```text
(v, q) → uniform sample (e.g. 2 fps, ≤512 frames)
       → single decode
       → parse Ŷ
```

| Effect | |
|--------|--|
| Long $T$, fixed $B$ | severe temporal downsample |
| Brief / late / repeated evidence | easy to skip |
| Same budget on all seconds | wastes tokens on empty bulk |

<span class="cite">Momentary evidence dies under sparse sample [Liu et al., 2026, Moment-Video].</span>

---

<!-- _header: Idea 1 · Niche -->
<!-- _footer: Chen et al., 2026; Zhang et al., 2026 -->

# Research niche

| Line | Needs train? | Our difference |
|------|--------------|----------------|
| OmniReasoner zoom tool | SFT+RL | frozen grounder, no tool policy |
| ScanFocus coarse→fine | specialized train | generalist VTG + matched $B$ |
| VideoTreeSearch | trained tree | shallow 2-stage, no RL nav |
| **Ours** | **no** | heuristic schedule + frozen $f$ |

<div class="formula" style="font-size:20px">

Keep strong open grounder. Redesign **where frames go**.

</div>

<span class="cite">OmniReasoner [Chen et al., 2026, arXiv:2607.19339]; ScanFocus [Chen et al., 2026, arXiv:2607.13421]; VTS [Zhang et al., 2026, arXiv:2607.16189].</span>

---

<!-- _header: Idea 1 · Problem -->
<!-- _footer: Zhu et al., 2026, §1 -->

# Formal setup

**Task**

$$
(v, q) \mapsto \mathcal{Y} = \{[s_k, e_k]\}_{k=1}^{K}
$$

**Baseline** $f_0$ — one-shot uniform

$$
\hat{\mathcal{Y}}_0 = f_0\!\left(v, q;\; \mathrm{SampleUniform}(v, B)\right)
$$

**Proposed** $f_H$ — hierarchical, **same** budget $B$

$$
\hat{\mathcal{Y}}_H = f_H(v, q;\, B)
$$

Success: higher set metrics at equal compute, or better Pareto.

---

<!-- _header: Idea 1 · Design -->
<!-- _footer: Zhu et al., 2026, §3.1 -->

# Design principle

Mirror **TimeLens2-93K logic** at test time — one frozen localizer.

| Data pipeline | Test-time analogue |
|---------------|-------------------|
| Coarse proposals $\mathcal{P}(q)$ | candidate windows |
| Local dual-agent ground | dense re-ground on survivors |
| Semantic check | embedding router (v1) |
| ±3s boundary refine | optional polish |
| Gap-merge ~1s | global set merge |

<span class="cite">Cascade structure from [Zhu et al., 2026, §3.1]; no weight updates for core method.</span>

---

<!-- _header: Idea 1 · Method -->
<!-- _footer: v1 policy · idea1 §1, §4 -->

# Pipeline overview

```text
video v + query q
        │
        ▼
┌───────────────────────────┐
│ Stage 1  RECALL           │  cheap scan · top-K windows
│  windows + embedding rank │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Stage 2  PRECISION        │  dense frames · frozen TimeLens2
│  local ground each W_(i)  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Stage 3  MERGE            │  global time · NMS · gap-merge
└─────────────┬─────────────┘
              ▼
         Ŷ_H  (interval set)
```

---

<!-- _header: Idea 1 · Method -->
<!-- _footer: Zhu et al., 2026, §3.1; Li et al., 2026 (Qwen3-VL-Embedding) -->

# Stage 1 · candidate discovery

**Goal:** recall-first — where *might* evidence live?

```text
windows:  scene cuts 20–60s  (+ 2s overlap)
          fallback: 45s window, 4s overlap
```

**v1 router** (not the full annotation ensemble):

| Step | Detail |
|------|--------|
| Sample | 4 sparse frames / window |
| Encode | Qwen3-VL-Embedding-2B + query |
| Rank | cosine similarity |
| Keep | $K = \min\!\big(8,\max(2,\lceil\sqrt{N}\rceil)\big)$ |

<span class="cite">Window spirit from TimeLens2 caption clips [Zhu et al., 2026, §3.1]. Embedding family used in their semantic check; here = **test-time router only**.</span>

---

<!-- _header: Idea 1 · Method -->
<!-- _footer: idea1 §4.4–4.6 -->

# Stage 2–3 · densify and merge

**Stage 2** — precision-first

```text
for each survivor W_(i):
    denser frames inside W_(i)
    Ŷ^(i) ← frozen TimeLens2-4B(W_(i), q)
    map clip times → global seconds
```

Allow empty set or multi-span per window.

**Stage 3** — set assembly

| Rule | |
|------|--|
| High-tIoU duplicates | suppress |
| Gap ≤ 1s | merge (jitter) |
| Disjoint hits | keep (multi-span) |

<span class="cite">Gap-merge ~1s follows annotation practice [Zhu et al., 2026, §3.1]. Absolute-time mapping (cf. TimeAnchor spirit [Chen et al., 2026]).</span>

---

<!-- _header: Idea 1 · Method -->
<!-- _footer: idea1 §4.7 -->

# Pseudocode (compressed)

```text
Input: v, q, frozen f, budget B
windows ← BuildWindows(v)                 # 20–60s / overlap
scores  ← EmbedRank(windows, q)           # 4 frames each
C       ← TopK(scores, K)

locals ← []
for W in C:
    y ← f(DenseSample(W), q)              # TimeLens2-4B
    locals.append(MapToGlobal(y, W))

return MergeSets(locals, gap=1s)
```

Router frames **count** toward proposed-method budget.

---

<!-- _header: Idea 1 · Intuition -->
<!-- _footer: Liu et al., 2026; Zhu et al., 2026 -->

# Why this can work

```text
uniform one-shot     ● · ● · ● · ● · ● · ●   (thin everywhere)

ours stage-1         · · · · ▲ · · · ▲ · ·   (find candidates)
ours stage-2         · · · · ███ · · ███ ·   (dense only there)
```

1. Evidence **sparse** — uniform wastes $B$ on empty bulk  
2. Labels already proved **local** decisions more reliable  
3. TimeLens2 is strong **local** grounder after SFT+GRPO  
4. Top-$K$ disjoint windows = natural **multi-span** support  

<span class="cite">[Liu et al., 2026]; label cascade [Zhu et al., 2026, §3.1].</span>

---

<!-- _header: Idea 1 · Controls -->
<!-- _footer: idea1 §1, §4.8, §7 -->

# What we must beat (honesty)

Not “call model more.” Beat **matched schedules**.

| ID | Schedule | Tests |
|----|----------|-------|
| A0 | **Uniform one-shot** | official-style baseline |
| A1 | **Full-video multipass** | more calls, no focus |
| A2 | **Uniform-window local** | local calls, no semantic route |
| A3 | **Embedding-window local** | **ours** |
| A4 | Offline oracle router | is router the bottleneck? |

```text
win only with more frames  →  claim fails
win at equal compute       →  claim holds
```

---

<!-- _header: Idea 1 · Budget -->
<!-- _footer: idea1 §1, §7.3 -->

# Compute accounting

**Primary unit:** decoded frames (+ report GPU time, calls, VRAM)

| Tier (nominal) | Example |
|----------------|---------|
| Small | 32 frames |
| Medium | 64 frames |

**Matched** = within **5%** aggregate GPU time  
else report **EtF1–compute Pareto** only.

v1 charges **router frames** to the proposed method.

---

<!-- _header: Idea 1 · Experiments -->
<!-- _footer: idea1 §1, §7 · revised 2026-07-23 -->

# Experimental goal (narrow)

> On official **OMTG Bench** (320 queries),  
> does embedding-routed local TimeLens2-4B  
> beat three matched controls — frozen weights, one T4?

| | |
|--|--|
| Why OMTG | 2–20 GT spans / query · set cardinality |
| Primary metric | **EtF1** |
| Secondary | C-Acc, tF1@0.3/0.5/0.7, union tIoU, card. error, router recall |
| Optional later | QVHighlights regression; VUE-TR-V2 if reproducible |

<span class="cite">OMTG stresses multi-span pain that motivated TimeLens2-93K. 93K = train source, not eval bench.</span>

---

<!-- _header: Idea 1 · Models -->
<!-- _footer: idea1 §7.1 -->

# Frozen stack (v1)

| Role | Model |
|------|--------|
| Local grounder | `MCG-NJU/TimeLens2-4B` |
| Stage-1 router | `Qwen/Qwen3-VL-Embedding-2B` |
| Hardware | 1× Colab T4 (models phased, not co-resident) |

**No** SFT. **No** GRPO. **No** learned zoom policy.

---

<!-- _header: Idea 1 · Positioning -->
<!-- _footer: Zhu et al., 2026; Chen et al., 2026; Zhang et al., 2026 -->

# Positioning sentence

> Prior generalist VTG-MLLMs improve **supervision and RL rewards** [Zhu et al., 2026].  
> Prior long-video agents improve **learned zoom / navigation** [Chen et al., 2026; Zhang et al., 2026].  
> We study **training-free hierarchical frame allocation** that reuses a **frozen** generalist grounder under **matched** inference budgets.

---

<!-- _header: Idea 1 · Risks -->
<!-- _footer: idea1 §8 -->

# Main risks

| Risk | Mitigation |
|------|------------|
| Stage-1 miss = fatal | overlap · larger $K$ · oracle ablation A4 |
| “Just slower” | matched GPU time + Pareto |
| Timestamp bugs | absolute seconds only |
| Short-video noise | disable hierarchy when $T$ small |
| Double-count spans | gap-merge + temporal NMS |

---

<!-- _header: Idea 1 · Scope -->
<!-- _footer: idea1 §9 -->

# Scope

| Is | Is not (v1) |
|----|-------------|
| Inference hierarchical search | New SFT / GRPO / reward |
| Training-free wrapper | Learned tool policy (OmniReasoner) |
| Matched-budget science | Full temporal tree RL (VTS) |
| Multi-span set merge | Spatiotemporal tubes |
| | Idea 2 (TW consensus) — later optional |

---

<!-- _header: Idea 1 · Success -->
<!-- _footer: idea1 §7.5 -->

# Success criteria

| Outcome | Meaning |
|---------|---------|
| **Strong** | higher EtF1 at matched GPU time; A1, A2 &lt; A3 |
| **Acceptable** | better Pareto, no exact matched win |
| **Negative but useful** | equal-$B$ gain vanishes; oracle A4 helps → router is bottleneck |

Scientific core ≠ “call twice.”  
Core = **reallocate frames toward sparse evidence**.

---

<!-- _header: Idea 1 · Abstract seed -->
<!-- _footer: idea1 §11 -->

# Abstract seed

> Generalist temporal grounding MLLMs such as TimeLens2 improve supervision and RL rewards, yet still localize with a **single forward pass under uniform frame sampling**. This schedule is misaligned with sparse long-video evidence and with the multi-stage search used to build their own labels. We propose a **training-free hierarchical** procedure that splits a fixed budget between **coarse candidate discovery** and **local dense re-grounding** with a frozen grounder, then merges variable-cardinality interval sets. Under matched compute, hierarchical allocation targets long-form / multi-span grounding without fine-tuning.

---

<!-- _header: References -->
<!-- _footer: "" -->

# Key references

<div class="quiet" style="font-size:15px; line-height:1.45; text-align:left">

**Zhu et al. (2026).** TimeLens2. arXiv:2607.17423.

**Liu et al. (2026).** Moment-Video. arXiv:2606.02522.

**Chen et al. (2026).** OmniReasoner. arXiv:2607.19339.

**Chen et al. (2026).** ScanFocus. arXiv:2607.13421 (ECCV).

**Zhang et al. (2026).** VideoTreeSearch. arXiv:2607.16189.

**Feng et al. (2026).** MoD-VLLM. arXiv:2607.15778.

**Li et al. (2026).** Qwen3-VL-Embedding (router family).

**Gao et al. (2017); Krishna et al. (2017).** Classical multi-scale / moment retrieval lineage.

</div>

---

<!-- _class: center -->
<!-- _footer: "" -->
<!-- _paginate: false -->

# 

<span class="quiet">Idea 1 · hierarchical test-time search</span>

<br>

<span class="cite">
Note: <code>Temporal_Grounding/idea1_hierarchical_test_time_search.md</code><br>
Code: <code>TimeLens2/evaluation/run_omtg_search.py</code><br>
Next: smoke OMTG first-25 · then full 320
</span>
