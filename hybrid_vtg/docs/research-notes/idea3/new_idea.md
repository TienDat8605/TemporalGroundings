
```md
# Idea 3: Scout-Guided Dense Evidence Grounding

## Summary

Scout-Guided Dense Evidence Grounding (SGDE) separates long-video temporal grounding into:

1. **Cheap global scouting** to find visually relevant time regions.
2. **Dense local LVLM grounding** to verify events and locate exact boundaries.

The main LVLM should not spend most of its visual-token budget searching irrelevant portions of a long video. Instead, a lightweight image-text scout scores frames densely across the full timeline, a deterministic temporal policy converts those scores into candidate windows, and the main LVLM receives dense before/during/after evidence only around the best candidates.

```text
Full video
  → image-text scout relevance timeline
  → temporal candidate proposals
  → dense local evidence packs
  → main LVLM verification and temporal grounding
  → validated source-video spans
```

## Motivation

Uniformly sampling a long video creates a trade-off:

- broad global coverage gives poor local boundary resolution;
- dense local sampling cannot cover the whole video.

SGDE resolves this by using a cheap scout for broad coverage and reserving the expensive main LVLM for high-value candidate regions.

The scout is optimized for **proposal recall**, not exact event boundaries.

## Models

### Scout

Primary candidate:

```text
google/siglip2-base-patch16-224
```

SigLIP2 is used as a frozen image-text retrieval model:

\[
s(t) = f_{\text{SigLIP2}}(\text{frame at }t, \text{query})
\]

It has no temporal reasoning requirement. Its role is to identify frames or short regions visually compatible with the query.

Alternative or comparison scout:

```text
Qwen/Qwen3-VL-Embedding-2B
```

This should be evaluated under the same sampling and proposal policy. Historical project results suggest Qwen embedding retrieval can be useful for window ranking, while the existing SigLIP2 route was weak on sparse 0.5-FPS TACoS cooking retrieval.

### Main grounder

```text
Qwen/Qwen3-VL-4B-Instruct
```

The frozen Qwen LVLM receives dense local frame sequences with timestamps and decides:

- whether the complete event actually occurs;
- whether the scout candidate is a false positive;
- how many occurrences exist;
- precise event start and end;
- confidence and boundary uncertainty.

## Scout Timeline

Sample the complete video at 0.5–2 FPS and compute query-conditioned scores:

\[
s(t) = f_{\text{scout}}(x_t, q)
\]

Scores are relative retrieval values, not calibrated event probabilities. Normalize within the current video:

\[
z(t) =
\frac{s(t) - \operatorname{median}(s)}
{\operatorname{MAD}(s) + \epsilon}
\]

Use conservative smoothing, preferably shot-aware, to reduce isolated noise without removing short events.

## Candidate Proposal Extraction

Do not maximize raw score mass:

\[
\max_{a,b}\int_a^b s(t)\,dt
\]

because it systematically favors long intervals.

Instead, generate candidates through several complementary mechanisms:

1. **Thresholded connected components**
   - Find connected time regions above a relevance threshold.
   - Use hysteresis: a high threshold enters an event and a lower threshold maintains it.

2. **Penalized interval scoring**

\[
J(a,b) =
\int_a^b (z(t)-\tau)\,dt - \lambda(b-a)
\]

This rewards sustained relevance but penalizes over-long windows.

3. **Multi-scale density windows**
   - Find local score peaks at several durations, such as 2, 5, 10, 20, and 40 seconds.
   - This supports both short and long events.

Rank candidates by peak score, average score, evidence mass, duration penalty, and query-component agreement. Keep the top-\(K\) temporally diverse windows with temporal NMS.

## Query Expansion

Complex queries may not map cleanly to one image. Generate retrieval subqueries for visual anchors.

Example:

```text
Full query:
  person starts riding a bicycle after picking up a helmet

Scout subqueries:
  person picking up a helmet
  person holding a helmet
  person riding a bicycle
  person with bicycle and helmet
```

The scout retrieves potential evidence. The main LVLM verifies the complete temporal relation.

## Dense Candidate Evidence

For every selected candidate, include:

```text
before context → estimated onset → event interior → estimated offset → after context
```

Candidate windows are padded adaptively. A minimum 2–5 seconds of left and right context is required, with greater padding for diffuse or uncertain scout evidence.

The main LVLM receives:

- a small set of global anchor frames;
- candidate frames sampled densely;
- additional frames near likely start/end regions;
- absolute source-video timestamps;
- explicit temporal roles such as `pre-context`, `candidate`, and `post-context`.

## Main-LVLM Grounding

The main LVLM is prompted to:

1. verify that the complete query is visually supported;
2. reject false-positive candidates;
3. identify all visually distinct occurrences;
4. compare evidence before, during, and after each event;
5. return source-video start/end times only when supported by evidence.

The LVLM should return structured predictions including event presence, temporal spans, confidence, and supporting frame timestamps.

## Boundary Refinement

After approximate LVLM localization, boundaries can be refined by locally sampling at higher FPS.

For starts, search for:

```text
absent → present
```

For ends, search for:

```text
present → absent
```

Maintain boundary corridors internally rather than assuming exact frame-level certainty.

## Global Rescue

The scout can miss an event. Preserve a fixed rescue budget for:

- global uniform anchors;
- scene-change frames;
- diverse low-confidence windows;
- fallback full-video or broad-evidence LVLM grounding if all candidates fail verification.

This prevents the scout from becoming an unrecoverable bottleneck.

## Evaluation

The primary scout metric is candidate coverage, not final IoU:

\[
\text{BoundaryCoverage@K} =
\mathbb{1}[
\exists c \in \text{TopK candidates}:
s_{\mathrm{GT}}, e_{\mathrm{GT}} \in c
]
\]

Evaluate:

- proposal recall and boundary coverage at top-\(K\);
- final temporal IoU and mAP;
- start/end boundary error;
- main-LVLM calls and visual-token cost;
- latency and memory;
- performance by event duration and query type.

Required comparisons:

1. uniform Qwen sampling;
2. current BGS;
3. SigLIP2 scout;
4. Qwen embedding scout;
5. scout-only proposals;
6. scout plus local re-scoring;
7. candidate-only evidence;
8. candidate evidence plus global anchors.

## Expected Contribution

SGDE changes the LVLM’s role from expensive global search to rigorous local temporal verification.

```text
Scout: where might useful evidence exist?
Main LVLM: does the full event occur, and exactly when?
```

The method is training-free, model-modular, budget-aware, and designed to improve boundary evidence without requiring the scout itself to have temporal reasoning capability.
```