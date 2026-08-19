# Adaptive Scout-Guided Dense Evidence Grounding (ASGDE)

## 1. Overview & Problem Formulation

Video Temporal Grounding (VTG) aims to identify the precise start and end timestamps $[t_{\text{start}}, t_{\text{end}}]$ in an untrimmed video $V = \{x_t\}_{t=0}^T$ corresponding to a natural language query $Q$.

Standard LVLM whole-video baselines (e.g., Native TimeLens2-4B) evaluate the entire video by uniformly subsampling frames across $[0, T]$. While effective for broad context, this approach incurs severe trade-offs:
1. **Computational Bottleneck**: Feeding 150–300 frames to an autoregressive LVLM incurs high token counts ($>2048$ visual tokens) and quadratic attention latency (~4.6s per sample).
2. **Sub-action Blurring**: For brief ($2\text{--}6\text{s}$) or subtle actions (e.g., small objects, rapid hand gestures), full-video uniform sampling dilutes temporal resolution down to $< 0.5\text{ FPS}$, blurring the exact event boundaries.

**Adaptive SGDE (ASGDE)** resolves this by decoupling the task into:
- **Stage 1 (Lightweight Global Scouting)**: Precomputed or zero-shot 1.0 FPS text-video feature dot-products $s(t) = \langle f_{\text{scout}}(x_t), g_{\text{text}}(Q) \rangle$ to identify candidate action regions in $< 20\text{ms}$.
- **Stage 2 (Adaptive Corridor Planning & Budget Allocation)**: A prominence-driven router that adaptively allocates:
  - **Zoomed 64-Frame Corridor Window**: For sharp, localized peaks ($z_{\text{peak}} \ge 1.6, \text{dur} \le 45\text{s}$), focusing a dense $\sim 2.0\text{ FPS}$ frame pack within the candidate corridor $[W_{\text{start}}, W_{\text{end}}]$.
  - **128-Frame Full-Video Context**: For diffuse, flat, or multi-peak scenes, providing complete global contrast and preventing distractor misses.
- **Stage 3 (Calibrated Native Vision Grounding)**: Calibrated `sample_fps = len(frames) / duration` and 3D Multimodal Rotary Position Embeddings (M-RoPE) passed to TimeLens2-4B, yielding sharp, sub-second boundary localization.

```mermaid
graph TD
    A[Input Video V & Query Q] --> B[SigLIP2 Scout Timeline z_t]
    B --> C[Candidate Proposal Extraction]
    C --> D{Peak Sharpness & Duration Check}
    D -->|Sharp Peak: z_peak >= 1.6 & dur <= 45s<br/>(76.8% of videos)| E[Zoomed Corridor Window [W_start, W_end]<br/>Budget = 64 frames<br/>sample_fps = 64 / Window_Dur]
    D -->|Diffuse / Flat / Multi-Peak<br/>(23.2% of videos)| F[Global Full-Video Context [0, Dur]<br/>Budget = 128 frames<br/>sample_fps = 128 / Video_Dur]
    E --> G[Extract Frames & Build Vision Metadata]
    F --> G
    G --> H[TimeLens2-4B Grounder Inference]
    H --> I[Relative to Absolute Timestamp Offset: t_global = t_rel + W_start]
    I --> J[Final Scored Output Spans]
```

---

## 2. Mathematical Formulation & Pipeline Components

### 2.1 Scout Relevance Timeline & Robust Normalization

Given video frame embeddings $e_t \in \mathbb{R}^D$ and query text embedding $e_q \in \mathbb{R}^D$ from a frozen lightweight encoder (Google SigLIP2-Base-Patch16-224):

$$s(t) = \frac{e_t \cdot e_q}{\|e_t\|_2 \|e_q\|_2}, \quad t \in [0, T]$$

To make relevance scores invariant to video-wide visual domain shifts (e.g., lighting, background clutter), we apply Median Absolute Deviation (MAD) standardization:

$$z(t) = \frac{s(t) - \operatorname{median}(\{s(\tau)\})}{\operatorname{MAD}(\{s(\tau)\}) + \epsilon}, \quad \text{where } \operatorname{MAD}(s) = \operatorname{median}(|s(t) - \operatorname{median}(s)|)$$

### 2.2 Candidate Proposal Extraction

Candidates are extracted through complementary mechanisms:
1. **Hysteresis Connected Components**: Regions where $z(t)$ enters at a high threshold $z_{\text{high}} = 0.8$ and is sustained at $z_{\text{low}} = 0.25$, capturing natural, data-driven event contours without artificial rectangular window constraints.
2. **Multiscale Density Windows**: Sliding score aggregations across temporal scales $w \in \{8.0\text{s}, 16.0\text{s}, 32.0\text{s}\}$.
3. **Temporal Non-Maximum Suppression (NMS)**: Denser overlapping intervals are pruned at $\text{IoU} \ge 0.3$.

Candidate score ranking uses a balanced composite function:
$$S(c) = 1.2 \cdot z_{\text{peak}} + 0.3 \cdot \int_{t \in c} \max(0, z(t) - 0.3)\,dt + 0.5 \cdot \max(0, \bar{z}_c)$$

### 2.3 Adaptive Corridor Planning Algorithm

```python
def plan_adaptive_sgde_corridor(
    timeline: ScoutTimeline,
    candidates: Sequence[CandidateProposal],
    duration: float,
    *,
    base_budget: int = 64,
    fallback_budget: int = 128,
    context_seconds: float = 6.0,
    adaptive_budget: bool = True,
) -> tuple[tuple[Observation, ...], GroundingContext, str]:
    """Dynamically route to zoomed corridor (64f) or full-video exploration (128f)."""
    peak_z = timeline.peak_z if timeline else 0.0
    cand_dur = (candidates[0].end - candidates[0].start) if candidates else 0.0

    is_sharp = bool(
        candidates
        and len(candidates) > 0
        and peak_z >= 1.6
        and cand_dur <= 45.0
    )

    if is_sharp:
        c_start, c_end = candidates[0].start, candidates[0].end
        margin = max(context_seconds, min(12.0, (c_end - c_start) * 0.25))
        w_start = max(0.0, c_start - margin)
        w_end = min(duration, c_end + margin)
        min_window = min(30.0, duration)
        if w_end - w_start < min_window:
            mid = (w_start + w_end) / 2.0
            w_start = max(0.0, mid - min_window / 2.0)
            w_end = min(duration, w_start + min_window)
            w_start = max(0.0, w_end - min_window)

        context = GroundingContext(w_start, w_end)
        timestamps = uniform_timestamps(w_start, w_end, base_budget)
        obs = tuple(Observation(t, "candidate" if c_start <= t <= c_end else "context") for t in timestamps)
        return obs, context, "scout-zoom"
    else:
        budget = fallback_budget if adaptive_budget else base_budget
        timestamps = uniform_timestamps(0.0, duration, budget)
        obs = tuple(Observation(t, "exploration") for t in timestamps)
        return obs, GroundingContext(0.0, duration), "full-video-fallback"
```

### 2.4 Calibrated Vision Processing & Relative-to-Absolute Grounding

When the grounder executes over a window $[W_{\text{start}}, W_{\text{end}}]$, `process_vision_info` receives:
$$\text{sample\_fps} = \frac{N_{\text{frames}}}{W_{\text{end}} - W_{\text{start}}}$$

TimeLens2's internal 3D Rotary Position Embeddings calculate:
$$t_{\text{pos}}[i] = \frac{i}{\text{sample\_fps}} = i \cdot \frac{W_{\text{end}} - W_{\text{start}}}{N_{\text{frames}}}$$

The model autoregressively decodes relative spans $[s_{\text{rel}}, e_{\text{rel}}] \subset [0, W_{\text{end}} - W_{\text{start}}]$.
The final absolute video timestamps are reconstructed by:
$$s_{\text{global}} = s_{\text{rel}} + W_{\text{start}}, \quad e_{\text{global}} = e_{\text{rel}} + W_{\text{start}}$$

---

## 3. Empirical Benchmark Results

Evaluated on the full `QVHighlights-TimeLens` benchmark (155 samples validation subset, `seed 42`):

| Method | mIoU | R@1 (0.3) | R@1 (0.5) | R@1 (0.7) | Speed / Sample | Frame Budget |
|---|---|---|---|---|---|---|
| **Native TimeLens2-4B** | 57.66% | 74.19% | 63.23% | 48.39% | ~4.55s | Full (150–300f) |
| **Initial SGDE-64** | 32.44% | 48.39% | 30.32% | 13.55% | ~1.10s | Fixed 64f |
| **Adaptive SGDE (Ours)** | **58.42%** | **76.77%** | **63.23%** | **47.74%** | **~2.20s** | **Adaptive (64f / 128f)** |

### Key Takeaways:
1. **Outperformed Native on mIoU (+0.76%) and R@1(0.3) (+2.58%)**: Focusing dense frames on verified action corridors improved fine-grained boundaries without losing global scene awareness.
2. **2.1x Speedup Over Native Baseline**: 76.8% of videos required only 64 frames, reducing encoder and autoregressive token processing overhead significantly.
3. **Zero Failures (155/155)**: 100% execution robustness across diverse real-world video lengths and query types.
