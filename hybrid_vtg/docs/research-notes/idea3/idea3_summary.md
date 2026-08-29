# Proposal Summary: Scout-Guided Dense Evidence Grounding (name still pending)

## Concept

Scout-Guided Dense Evidence Grounding (SGDE) separates long-video temporal grounding into a cheap global search followed by dense local verification:

1. Sample the complete video with a frozen image-text scout.
2. Compute a query-conditioned relevance timeline.
3. Normalize and smooth the timeline within each video.
4. Extract and rank candidate temporal regions.
5. Send dense evidence from the best region to the main LVLM.
6. Convert the model's result back to absolute source-video timestamps.

The scout is intended to maximize proposal recall, not to determine exact event boundaries. This preserves the main model's visual-token budget for relevant before, during, and after evidence.

## Proposed Design

The proposal is a training-free, model-modular pipeline using:

- **Scout:** `google/siglip2-base-patch16-224`, with `Nemotron-1B` as a comparison option.
- **Grounder:** Qwen/Qwen3-VL-4B-Instruct / Timelens2-4B, which also derived from Qwen3-VL.
- **Timeline:** cosine image-text scores, conservative smoothing, and robust median/MAD normalization.
- **Proposals:** hysteresis connected components, penalized intervals, multi-scale density windows, query-component agreement, and temporal NMS.
- **Evidence:** adaptive padding, explicit pre-context/candidate/post-context roles, global rescue anchors, and denser sampling near approximate boundaries.
- **Verification:** structured event presence, occurrence count, spans, confidence, and uncertainty, followed by local boundary refinement.

The design also calls for a rescue budget and a fallback full-video path so a scout miss does not become an unrecoverable false negative.

### Scout
- cached or on-demand video and query embeddings;
- default 1 FPS timeline sampling;
- cosine relevance scores;
- triangular moving-average smoothing;
- per-video robust median/MAD normalization;
- peak-z and cache/model telemetry.

### Candidate extraction

- hysteresis components with high and low thresholds;
- penalized interval scoring;
- multi-scale density windows;
- composite ranking using peak, positive evidence mass, and mean relevance;
- temporal NMS with deterministic ordering and cardinality-dependent limits.