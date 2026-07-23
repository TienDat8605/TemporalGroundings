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
    padding: 48px 56px 56px 56px;
  }
  h1 {
    font-size: 34px;
    font-weight: 600;
    letter-spacing: -0.02em;
    border-bottom: none;
    margin: 0 0 0.55em 0;
  }
  table {
    font-size: 22px;
    width: 100%;
    border-collapse: collapse;
  }
  th {
    font-weight: 600;
    text-align: left;
    border-bottom: 1.5px solid #222;
    padding: 0.32em 0.45em;
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
    margin: 0.55em 0;
  }
  footer { font-size: 13px; color: #888; }
  header { font-size: 12px; color: #999; text-align: right; }
  .cite {
    font-size: 15px;
    color: #666;
    margin-top: 0.85em;
    line-height: 1.35;
  }
  .quiet { color: #666; font-size: 18px; }
  .tag {
    font-size: 13px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #777;
    margin-bottom: 0.35em;
  }
---

<!-- Three core pitch slides: train fix → inference gap → our idea + OMTG -->

<!-- _header: Prior work -->
<!-- _footer: Zhu et al., TimeLens2, arXiv:2607.17423 -->

# TimeLens2 fixed **training**

<div class="tag">Labels + reward · not inference</div>

| Fix | What it does |
|-----|----------------|
| **TimeLens2-93K** | Multi-stage labels: propose → local dual-agent → consensus → semantic check → boundary refine |
| **SFT → GRPO** | Learn search syntax, then calibrate geometry |
| **Reward** | $R = R_{\mathrm{tIoU}} + R_{\mathrm{TW}} - \mathbf{1}_{\mathrm{invalid}}$ |

```text
data:   multi-scale verification cascade
train:  set-valued interval geometry (match-free W₁)
```

<span class="cite">Treats evidence as an interval set through supervision and optimization [Zhu et al., 2026, §3].</span>

---

<!-- _header: Gap -->
<!-- _footer: Zhu et al., 2026; Liu et al., 2026 -->

# Gap: inference still **one-shot uniform**

```text
(v, q)  →  uniform frames over full T   (e.g. 2 fps, cap ~512)
        →  single forward decode
        →  parse Ŷ
```

| Train / data pipeline | Deployed inference |
|-----------------------|--------------------|
| Propose → verify local → refine | **One** pass |
| Dense pixels on short clips | Thin sample on **all** of $T$ |
| Multi-span built carefully | Easy to **skip** brief / late / repeated hits |

```text
irony:  labels made by hierarchical search
        model used without hierarchical search
```

<span class="cite">Sparse evidence fails under uniform downsample [Liu et al., 2026]. Adaptive test-time allocation not first-class in TimeLens2 [Zhu et al., 2026].</span>

---

<!-- _header: This work -->
<!-- _footer: Idea 1 · OMTG primary eval -->

# Our idea · hierarchical test-time search

<div class="tag">Training-free · frozen TimeLens2-4B · matched budget</div>

```text
Stage 1  RECALL     window video · embed-rank top-K
Stage 2  PRECISION  dense frames only on survivors · local ground
Stage 3  MERGE      global timestamps · gap-merge · set Ŷ
```

**Not** new SFT/GRPO. **Reallocate** frames toward sparse multi-span evidence.

| Eval (primary) | |
|----------------|--|
| Bench | **OMTG** · 320 queries · 2–20 GT spans / query |
| Metric | **EtF1** (set-aware; cardinality matters) |
| Controls | one-shot · full multipass · uniform-window · **ours** |
| Rule | win only if better at **matched** compute (≤5% GPU time) |

<span class="cite">Router: Qwen3-VL-Embedding-2B. Grounder frozen. Optional later: QVHighlights regression.</span>
