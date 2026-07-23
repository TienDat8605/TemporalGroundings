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
    font-size: 34px;
    font-weight: 600;
    letter-spacing: -0.02em;
    border-bottom: none;
    margin: 0 0 0.55em 0;
  }
  h2 {
    font-size: 20px;
    font-weight: 500;
    color: #555;
    margin: 0 0 0.8em 0;
  }
  p, li { line-height: 1.4; }
  ul { padding-left: 1.1em; }
  table {
    font-size: 21px;
    width: 100%;
    border-collapse: collapse;
  }
  th {
    font-weight: 600;
    text-align: left;
    border-bottom: 1.5px solid #222;
    padding: 0.3em 0.45em;
  }
  td {
    border-bottom: 1px solid #e6e6e6;
    padding: 0.32em 0.45em;
    vertical-align: top;
  }
  code, pre {
    font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
    font-size: 17px;
    background: #f5f5f5;
  }
  pre {
    padding: 0.85em 1.05em;
    border-radius: 5px;
    border: 1px solid #e6e6e6;
    line-height: 1.5;
    margin: 0.6em 0;
  }
  strong { font-weight: 600; }
  em { font-style: italic; }
  footer {
    font-size: 13px;
    color: #888;
    font-family: "Helvetica Neue", sans-serif;
  }
  header {
    font-size: 12px;
    color: #999;
    text-align: right;
  }
  section.center {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .cite {
    font-size: 15px;
    color: #666;
    margin-top: 0.85em;
    line-height: 1.35;
  }
  .quiet {
    color: #666;
    font-size: 18px;
  }
  .formula {
    font-size: 28px;
    text-align: center;
    margin: 1.2em 0;
  }
  .label {
    font-size: 13px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #777;
    margin-bottom: 0.4em;
  }
---

<!--
Problem diagnosis deck (minimal, scientific, cited)
Primary source: Temporal_Grounding/2607.17423_why_existing_training_failed.md
-->

<!-- _class: center -->
<!-- _footer: "" -->
<!-- _paginate: false -->

# Why existing training failed
## for video temporal grounding

<span class="quiet">Problem diagnosis · not the method</span>

<br>

<span class="cite">
Primary: Zhu et al., <em>TimeLens2</em>, arXiv:2607.17423, 2026<br>
Supporting: Zhang et al. 2026; Shao et al. 2024; Luo et al. 2026; Liu et al. 2026
</span>

---

<!-- _header: Problem · Task -->
<!-- _footer: Zhu et al., 2026, §1 -->

# The task object

<div class="formula">

$(v,\, q) \;\mapsto\; \mathcal{Y} = \{[s_k,\, e_k]\}_{k=1}^{K}$

</div>

| Symbol | Meaning |
|--------|---------|
| $v$ | video |
| $q$ | language query |
| $\mathcal{Y}$ | **variable-cardinality** set of evidence intervals |

<span class="cite">Not free-form captioning — a mapping to an interval set [Zhu et al., 2026, §1].</span>

---

<!-- _header: Problem · Task -->
<!-- _footer: Zhu et al., 2026, §1 -->

# Citation, not description

Temporal grounding = video analogue of a **citation**.

| | Description (*what*) | Grounding (*when*) |
|--|----------------------|---------------------|
| Output | fluent narrative | precise intervals |
| Verifiable? | hard | audit on timeline |
| If missing | — | user scrubs full video |

<span class="cite">Without temporal support, even a correct description remains hard to verify [Zhu et al., 2026, §1].</span>

---

<!-- _header: Problem · Motivating example -->
<!-- _footer: cf. Liu et al., 2026 (Moment-Video); Zhu et al., 2026 -->

# Motivating example

<div class="label">Setting</div>

**12 min** basketball highlight  
Query: *When does the ball change direction after the block?*

```text
0:00 ────────────────────|█|──────────────── 12:00
                         ↑
                   ~0.3 s  @  09:41
```

Rest of reel: dribbles, crowd, similar blocks (distractors).

<span class="cite">Illustrative sparse-evidence case. Related failure mode: momentary events under sparse sampling [Liu et al., 2026].</span>

---

<!-- _header: Problem · Motivating example -->
<!-- _footer: Liu et al., 2026; Zhu et al., 2026, §1 -->

# Same video · two skills

```text
WHAT    "Defender blocks; the ball goes out."     ✓  fluent
WHEN    [ ???? ]  or wrong possession             ✗  not citable
```

| Skill | Needs | Failure mode |
|-------|-------|--------------|
| *What* | some relevant frames + language prior | still looks right |
| *When* | **those** frames + bounds | brief evidence skipped |

<span class="cite">Answer-critical evidence may last few frames; sparse sample / compression can erase it before language reasoning [Liu et al., 2026].</span>

---

<!-- _header: Problem · Motivating example -->
<!-- _footer: Liu et al., 2026 -->

# Why *what* can succeed while *when* fails

```text
uniform sample   ● · · ● · ● · · ● · ●
true evidence              [█]           ← missed

model still observes:  court · players · motion
language prior fills:  block → ball out
```

<div class="formula" style="font-size:22px">

$\mathrm{duration}(\text{evidence}) \ll T  \;\Rightarrow\;  P(\text{hit}\mid \text{uniform}, B) \text{ small}$

</div>

<span class="cite">Denser FPS helps some models but does not remove the bottleneck [Liu et al., 2026].</span>

---

<!-- _header: Problem · Scope -->
<!-- _footer: Gao et al., 2017; Hendricks et al., 2017; Krishna et al., 2017; Zhu et al., 2026 -->

# Classical vs generalist

| Axis | Classical bias | Generalist requirement |
|------|----------------|------------------------|
| Length | short / clipped | seconds inside min–hour |
| Cardinality $K$ | usually one | repeated / disjoint |
| Query | declarative | + questions |
| Viewpoint | third-person | + egocentric |
| Interface | loc. head | generative MLLM |

<span class="cite">Classical lineage: Charades-STA [Gao et al., 2017], DiDeMo [Hendricks et al., 2017], ActivityNet Captions [Krishna et al., 2017]. Generalist framing: [Zhu et al., 2026, §1].</span>

---

<!-- _header: Problem · Scope -->
<!-- _footer: Zhu et al., 2026, Abstract & §1 -->

# Interval set, not a box list

$$
\mathrm{merge}(\mathcal{A}) \;=\; \bigcup_{[s,e]\in\mathcal{A}} [s,e]
$$

```text
{[10, 30]}    ≡_support    {[10, 20], [20, 30]}
```

1. **Cardinality free** — $K \in \{0,1,2,\ldots\}$
2. **Partition ≠ identity** — same mass, different cuts

<span class="cite">Prior pipelines do not treat $\mathcal{Y}$ as a first-class set through supervision *and* optimization [Zhu et al., 2026].</span>

---

<!-- _header: Problem · Core claim -->
<!-- _footer: Zhu et al., 2026, §1 -->

# Two structural mismatches

```text
     task: set-valued evidence on long, distractor-heavy video
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      A · SUPERVISION           B · OPTIMIZATION
         labels                    loss / reward
             │                         │
      brittle · narrow           tokens · flat tIoU
```

<span class="cite">Progress blocked by misalignment, not backbone scale alone [Zhu et al., 2026, §1].</span>

---

<!-- _header: Mismatch A · Supervision -->
<!-- _footer: Gao et al., 2017; Hendricks et al., 2017; Krishna et al., 2017; Zhu et al., 2026, §2 -->

# A · Historical data prior

Early benchmarks: **short** video + **single** descriptive moment.

```text
train mixture prior  →  single-span · caption-style
                     ↛  multi-span search among distractors
```

| Under-practiced | Result |
|-----------------|--------|
| Evidence search | weak long-horizon transfer |
| Completeness ($K>1$) | missed repeats |
| Question / ego forms | coverage holes |

<span class="cite">Early shape: [Gao et al., 2017; Hendricks et al., 2017; Krishna et al., 2017]. Inheritance critique: [Zhu et al., 2026, §2].</span>

---

<!-- _header: Mismatch A · Supervision -->
<!-- _footer: Zhu et al., 2026, §1, §3.1 -->

# A · Long-video labels are brittle

One global annotation pass:

```text
··· similar action ···  true hit  ··· similar ···
         ↑ wrong occurrence
         ↑ or only first labeled
         ↑ or fuzzy start/end
```

| Failure mode | Mechanism |
|--------------|-----------|
| Wrong occurrence | distractor confused with true hit |
| Missed repeats | later supporting spans dropped |
| Imprecise bounds | sparse evidence, drift of seconds |

<span class="cite">Long-video annotation is an evidence-verification problem, not only a scaling cost [Zhu et al., 2026, §1, §3.1].</span>

---

<!-- _header: Mismatch A · Supervision -->
<!-- _footer: Zhu et al., 2026, Table 3; Zhang et al., 2026 -->

# A · Scale ≠ quality

```text
raw dual-agent labels     ~735K–999K     ~42.0 mIoU
         │  consensus + semantic filter
         ▼
verified set                 93.2K       improves
         │  boundary refine (same 93.2K)
         ▼
refined                      93.2K       +1.7 stage gain
```

Trustworthy $\mathcal{Y}$ matters more than more timestamp strings.

<span class="cite">Ablation: [Zhu et al., 2026, Table 3]. Label noise can change rankings: TimeLens [Zhang et al., 2026].</span>

---

<!-- _header: Mismatch A · Supervision -->
<!-- _footer: Yuan et al., 2026; Grauman et al., 2022; Vidi Team et al., 2025 -->

# A · Coverage holes remain

Even clean local labels leave transfer gaps:

| Gap | Example benchmark |
|-----|-------------------|
| Question-form | MomentSeeker [Yuan et al., 2026] |
| Egocentric | Ego4D-NLQ [Grauman et al., 2022] |
| Long multi-span | VUE-TR / VUE-TR-V2 [Vidi Team et al., 2025] |

Short declarative scores can hide generalist failure.

<span class="cite">Curriculum / coverage problem, not only loss design [Zhu et al., 2026, §1–2].</span>

---

<!-- _header: Mismatch B · Optimization -->
<!-- _footer: Ren et al., 2023; Huang et al., 2024; Zeng et al., 2025; Zhu et al., 2026, §2 -->

# B · SFT optimizes tokens

Generative grounding emits timestamp **text**.  
Loss = next-token cross-entropy.

| Learns | Does not learn |
|--------|----------------|
| answer format | temporal overlap |
| that times appear | 2 s vs 20 min geometry |
| surface patterns | long-context search |

<span class="cite">SFT teaches timestamp *syntax*, not evidence *quality* [Zhu et al., 2026, §2]. Lineage: TimeChat [Ren et al., 2023], VTimeLLM [Huang et al., 2024], TimeSuite [Zeng et al., 2025].</span>

---

<!-- _header: Mismatch B · Optimization -->
<!-- _footer: Shao et al., 2024; Wang et al., 2025; Chen et al., 2025; Zhu et al., 2026 -->

# B · RL + tIoU: better, incomplete

Set-level overlap on merged supports:

$$
R_{\mathrm{tIoU}}(\hat{\mathcal{Y}},\mathcal{Y})
=
\frac{\lvert\mathrm{merge}(\hat{\mathcal{Y}})\cap\mathrm{merge}(\mathcal{Y})\rvert}
     {\lvert\mathrm{merge}(\hat{\mathcal{Y}})\cup\mathrm{merge}(\mathcal{Y})\rvert}
$$

Aligns training with mIoU / R@$\mu$ — but only **after** intersection.

<span class="cite">GRPO [Shao et al., 2024]; VTG-RL e.g. Time-R1 [Wang et al., 2025], TVG-R1 [Chen et al., 2025]. Set form as in [Zhu et al., 2026, §3.2].</span>

---

<!-- _header: Mismatch B · Optimization -->
<!-- _footer: Zhu et al., 2026, §2, §3.2, Fig. 3a -->

# B · The tIoU plateau

```text
GT       ──────────[████████]──────────

near     ──────[██]────────────────────   tIoU = 0
far      ────────────────────[██]──────   tIoU = 0
```

**Near miss ≡ far miss.**  
No graded “shift right” vs “jump region.”

<span class="cite">Overlap supplies geometry only after prediction and target intersect [Zhu et al., 2026, §2, Fig. 3a].</span>

---

<!-- _header: Mismatch B · Optimization -->
<!-- _footer: Shao et al., 2024; Zhu et al., 2026, Table 9 -->

# B · GRPO goes silent

Group-relative learning needs **within-group** ranking.

```text
all G rollouts miss GT
    →  R_i = 0  ∀i
    →  A_i = R_i − mean(R) = 0
    →  no temporal update
```

| Diagnosis (valid groups) | Value |
|--------------------------|-------|
| Constant reward under $R_{\mathrm{tIoU}}$ alone | **13.8%** |
| All-zero-tIoU groups later rescued by distance term | **75.8%** |

<span class="cite">GRPO [Shao et al., 2024]. Numbers: [Zhu et al., 2026, Table 9].</span>

---

<!-- _header: Mismatch B · Optimization -->
<!-- _footer: Luo et al., 2026; Zhu et al., 2026, Fig. 3c–d -->

# B · Matching is a fragile set proxy

Multi-span patch: one-to-one match + pairwise NGIoU [Luo et al., 2026].

```text
same merged support, different cuts:

  Y₁ = {[100, 404]}
  Y₂ = {[100, 105], [105, 404]}

  matched score:  1.0  vs  0.32   (TimeLens2 illustration)
```

| Artifact | Effect |
|----------|--------|
| Fragmentation | fragment ↔ wrong target |
| Unequal $K$ | merge / extra spans distort assignment |
| Partition dependence | serialization scored, not mass |

<span class="cite">[Zhu et al., 2026, Fig. 3c–d]; MUSEG [Luo et al., 2026].</span>

---

<!-- _header: Problem · Synthesis -->
<!-- _footer: Zhu et al., 2026, §1–3 -->

# Failure cascade

```text
narrow / noisy interval labels
            │
            ▼
   SFT imitates timestamp text
            │
            ▼
   tIoU RL: zero on all misses  →  GRPO silent
            │
            ▼
   matched multi-span rewards  →  partition artifacts
            │
            ▼
   short-bench OK · long / multi / question / ego weak
```

<span class="cite">Both ends must treat evidence as an interval set [Zhu et al., 2026, §5].</span>

---

<!-- _header: Problem · Synthesis -->
<!-- _footer: Zhu et al., 2026, Fig. 3 (restated) -->

# Mis-ranking under old signals

**GT:** $[100,105] \cup [400,404]$

| Prediction $\hat{\mathcal{Y}}$ | Human quality | tIoU |
|--------------------------------|---------------|------|
| both spans exact | best | high |
| near miss on first only | close, incomplete | **0** |
| far miss in middle | bad | **0** |
| first event only | incomplete | mid / low |
| same support, different cuts | ~equivalent | match scores may **diverge** |

<span class="cite">Practical content of [Zhu et al., 2026, Fig. 3].</span>

---

<!-- _header: Problem · Claim -->
<!-- _footer: Zhu et al., 2026; Zhang et al., 2026; Ren et al., 2023; Shao et al., 2024; Luo et al., 2026 -->

# Condensed claim

**Training never consistently optimized the true task object.**

| Side | Proxy used | Object needed |
|------|------------|---------------|
| Data | one-pass / single-span corpora | verified interval **sets** |
| SFT | next-token syntax | geometry of $\mathcal{Y}$ |
| RL | overlap after hit; brittle matches | graded, set-level signal |

→ Models stay fluent at **what**, unreliable at **when**.

<span class="cite">[Zhu et al., 2026, §9 condensed]; data quality [Zhang et al., 2026]; SFT lineage [Ren et al., 2023]; GRPO [Shao et al., 2024]; matching [Luo et al., 2026].</span>

---

<!-- _header: Problem · Checklist -->
<!-- _footer: Diagnostic from Zhu et al., 2026 -->

# Reading checklist

When evaluating a new VTG-MLLM paper:

1. **Labels** — single box only? multi-span agreement checked?
2. **Context** — full-video search, or refine on cropped clips?
3. **SFT** — only next-token CE on timestamp text?
4. **RL** — zero-overlap rollouts distinguished?
5. **Sets** — reward via match, or merged support?
6. **Eval** — gains beyond short single-span benches?

<span class="cite">If (3)–(5) are weak, Charades-like scores may overstate generalist ability [Zhu et al., 2026].</span>

---

<!-- _header: References -->
<!-- _footer: "" -->

# References (1/2)

<div class="quiet" style="font-size:15px; line-height:1.45; text-align:left">

**Chen et al. (2025).** Datasets and recipes for video temporal grounding via reinforcement learning. arXiv:2507.18100.

**Gao et al. (2017).** TALL: Temporal activity localization via language query. *ICCV*.

**Grauman et al. (2022).** Ego4D: Around the world in 3,000 hours of egocentric video. *CVPR*.

**Hendricks et al. (2017).** Localizing moments in video with natural language. arXiv:1708.01641.

**Huang et al. (2024).** VTimeLLM: Empower LLM to grasp video moments. *CVPR*.

**Krishna et al. (2017).** Dense-captioning events in videos. *ICCV*.

**Liu et al. (2026).** Moment-Video: Diagnosing temporal fidelity of video MLLMs on momentary visual events. arXiv:2606.02522.

**Luo et al. (2026).** MUSEG: Reinforcing video temporal understanding via timestamp-aware multi-segment grounding. *ACL*.

</div>

---

<!-- _header: References -->
<!-- _footer: "" -->

# References (2/2)

<div class="quiet" style="font-size:15px; line-height:1.45; text-align:left">

**Ren et al. (2023).** TimeChat: A time-sensitive multimodal large language model for long video understanding. arXiv:2312.02051.

**Shao et al. (2024).** DeepSeekMath. arXiv:2402.03300. (GRPO)

**Vidi Team et al. (2025a,b).** Vidi / Vidi2. arXiv:2504.15681, arXiv:2511.19529.

**Wang et al. (2025).** Time-R1: Post-training large vision language model for temporal video grounding. arXiv:2503.13377.

**Yuan et al. (2026).** MomentSeeker: A task-oriented benchmark for long-video moment retrieval. *NeurIPS*.

**Zeng et al. (2025).** TimeSuite: Improving MLLMs for long video understanding via grounded tuning. *ICLR*.

**Zhang et al. (2026).** TimeLens: Rethinking video temporal grounding with multimodal LLMs. *CVPR*.

**Zhu et al. (2026).** TimeLens2: Generalist video temporal grounding with multimodal LLMs. arXiv:2607.17423.

</div>

---

<!-- _class: center -->
<!-- _footer: "" -->
<!-- _paginate: false -->

# 

<span class="quiet">End of problem diagnosis</span>

<br>

<span class="cite">
Full prose note: <code>Temporal_Grounding/2607.17423_why_existing_training_failed.md</code><br>
Next deck: inference gap · hierarchical test-time search
</span>
