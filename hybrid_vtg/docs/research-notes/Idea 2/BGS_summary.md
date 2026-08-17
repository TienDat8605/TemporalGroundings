# Boundary-Guided Sparsification (BGS)

## Purpose

Boundary-Guided Sparsification is a training-free temporal grounding policy that sits in front of an evidence-capable model such as Qwen. Its objective is to find event start and end times by spending frames where temporal state changes are likely, rather than asking a generative model to infer the full answer from a uniformly sampled video.

BGS treats temporal grounding as a staged process:

1. Locate coarse evidence of the query throughout the video.
2. Convert evidence strength into a timeline of absent, uncertain, and present states.
3. Detect persistent transitions between those states.
4. Form plausible event episodes from start/end transition pairs.
5. Spend the remaining sampling budget refining the boundaries of the best episodes.
6. Return deterministic BGS spans when the refined evidence is sufficient.
7. Use Qwen generation only as a constrained fallback when BGS cannot produce a viable span.

## Budget and Sampling Policy

BGS uses a duration-scaled, frame-counted budget. The budget consists of:

- One logical scout observation for every fixed 8-second video cell.
- A fixed 64-physical-frame post-scout reserve.
- Exact ledger accounting for all requested frames, including Qwen-specific duplicate tubelet frames and any even-frame padding.

The matched uniform-budget baseline is intentionally unchanged. BGS reallocates only its own fixed post-scout reserve.

For Qwen-style visual encoders, every logical observation is represented by two nearby physical frames. This preserves a meaningful temporal unit for Qwen's tubelet-based visual processing. BGS aggregates the two physical frames back into one logical observation before temporal reasoning. The same treatment applies to the scout, rescue observations, and boundary refinement.

## Coarse Presence Timeline

The initial scout samples the center of each 8-second cell. BGS scores every visual evidence row against the query and combines spatial rows at the same logical time using an upper-quartile mean. This reduces the effect of isolated high-scoring spatial regions while retaining strong evidence.

BGS normalizes scout evidence relative to the video itself using a robust median and median absolute deviation calibration. The resulting logical observations are classified as:

- Absent: low relative query presence.
- Present: high relative query presence.
- Uncertain: evidence between those two states, or a timeline with insufficient variation.

This calibration is global for the sample. Later rescue and refinement observations use the same calibration, so their states remain comparable to the scout rather than being rescaled locally.

## Adaptive Rescue Scout

Before episode detection, BGS can spend up to 16 physical frames from the existing 64-frame reserve on a denser rescue scout. It does not increase the total budget.

The rescue scout targets locations most likely to be missed by the fixed 8-second sampling grid:

- Isolated present cells, which may represent short events that lack the two consecutive observations needed for a persistent transition.
- Uncertain cells with relatively strong evidence.

Each selected 8-second cell is subdivided into two interior observations. Rescue observations are merged chronologically with the scout timeline before transition detection. This can convert a short, isolated indication into persistent temporal evidence while preserving the original global calibration.

Unused rescue allocation remains available for refinement. Candidate admission is always based on the ledger's actual remaining capacity after rescue sampling.

## Boundary and Episode Construction

BGS requires two consecutive confident observations in the same state to confirm a state change. This persistence requirement reduces reactions to single-frame or single-cell noise.

A confirmed absent-to-present transition becomes a start boundary bracket. A confirmed present-to-absent transition becomes an end boundary bracket. Video edges can act as natural start or end boundaries when an event begins or ends at the video boundary.

Start and end brackets are paired chronologically into candidate episodes. BGS ranks candidates by reliability instead of relying only on visual contrast. Reliability incorporates:

- Directional confidence of the start and end transitions.
- Strength of nearby witness states.
- The compactness of the interval.
- The proportion of confident-present observations inside the interval.
- Penalties for excessive interval length.

BGS rejects implausible pairs before refinement when they are too distant or contain a persistent, strong absent run inside the proposed episode. This prevents a start and an unrelated later end from being combined into one broad event.

For multi-occurrence queries, BGS keeps at most six non-overlapping candidates. For single-occurrence queries, it keeps one. Ties are deterministic.

## Boundary Refinement

For each admitted episode, BGS narrows the start and end brackets toward a 1-second target resolution. It samples bracket midpoints and keeps the side whose state agrees with the expected transition direction.

If a midpoint remains uncertain, BGS can make one directional quarter-probe round. A boundary becomes:

- Resolved when its remaining corridor is at or below the target resolution.
- Ambiguous when evidence remains uncertain or the refinement depth is exhausted.

Admission is conservative: BGS only selects candidates whose complete worst-case refinement fits in the post-rescue ledger. It does not partially refine a candidate and then discover that the budget is exhausted.

## Primary BGS Output

BGS returns its own primary temporal spans whenever at least one selected episode has usable, chronological start and end corridors. This path does not invoke generative Qwen prediction.

A resolved corridor contributes its midpoint as a normal boundary estimate. A bounded ambiguous corridor can also contribute its midpoint conservatively, provided its width is no greater than 8 seconds. The resulting span receives a penalty for each ambiguous boundary and for boundary width.

This preserves useful candidate episodes that have clear temporal support but cannot be refined to the 1-second goal. BGS excludes pairs with missing boundaries, non-chronological midpoint endpoints, or overly wide ambiguous corridors.

Primary-span provenance distinguishes:

- Resolved: both boundaries are resolved.
- Conservative start: only the start is ambiguous but bounded.
- Conservative end: only the end is ambiguous but bounded.
- Conservative both: both boundaries are ambiguous but bounded.

The primary output is a JSON list of absolute source-video spans.

## Qwen Fallback

Qwen generation is a fallback rather than the default BGS output. It is used only when BGS has no viable primary span, such as when there are no candidates, none fit the refinement budget, or all refined corridors are ineligible.

Before fallback generation, BGS can use remaining frames for supplementary evidence. It preserves broad global coverage and allocates additional observations inside selected candidate episodes where possible. Evidence is tagged by temporal role, including global anchors, event interiors, resolved boundaries, ambiguous boundaries, and rescue observations.

If the model has a maximum evidence-unit capacity, BGS compacts the merged evidence. It protects left and right witnesses around candidate corridors first, then retains score-ranked temporal anchors. This keeps boundary evidence available to the fallback model while respecting its input limit.

Fallback is constrained when BGS has valid candidate support:

- BGS builds padded support windows from top-ranked, non-overlapping selected candidates. If none were selected, it uses ranked valid candidates.
- Qwen receives the candidate windows and an expected maximum occurrence count in its grounding context.
- The Qwen prompt tells it to keep returned intervals within those windows and not exceed the occurrence cap.
- BGS independently enforces the constraint after generation. It clips each span to its best-overlap support window, drops spans with no overlap, applies temporal non-maximum suppression, and caps the result to the number of support windows.

If BGS has no support windows, fallback remains unconstrained to avoid suppressing an event that the scout failed to identify.

## Differences from Direct Qwen Grounding

Direct Qwen grounding encodes supplied evidence and generates timestamp spans from a natural-language prompt. It relies on the model to interpret the temporal evidence, enumerate occurrences, and decide the final intervals.

BGS changes that behavior in several important ways:

- BGS is a sampling and temporal-reasoning policy, not a new trained model.
- It uses query-to-visual relevance scores to construct a calibrated absent/present timeline before generation.
- It adapts sampling toward transition evidence instead of treating all frames as equally useful.
- It enforces a duration-matched physical-frame ledger and explicitly accounts for Qwen tubelets.
- It handles short-event recovery through a limited adaptive rescue scout.
- It identifies, rejects, ranks, and admits event candidates before spending refinement frames.
- It can produce deterministic boundary-derived spans without any Qwen generation call.
- It allows bounded uncertainty to survive as a lower-confidence conservative span instead of converting every ambiguous boundary into total fallback.
- When Qwen fallback is necessary, BGS gives it candidate-window and occurrence-count guidance, then validates and filters its output against the same support.
- It retains an unconstrained Qwen path only when BGS has no defensible candidate support.

Qwen remains responsible for visual encoding, query scoring, and generative fallback. BGS supplies the temporal policy, budget allocation, boundary logic, output eligibility, and post-generation constraints.

## Qwen Compatibility and Numerical Handling

Qwen's visual encoder can use bfloat16 while its processor emits float32 pixel tensors. The evidence encoder aligns processor pixels to the vision tower's parameter dtype before visual feature extraction. This prevents float/bfloat16 mismatch failures during BGS's multiple encode stages as well as ordinary Qwen evidence encoding.

BGS preserves absolute source-video timestamps throughout sampling and evidence assembly. Qwen receives relative timestamps in its prompt according to the grounding context, while BGS converts and evaluates spans in source-video time.

## Observability

BGS records detailed telemetry for diagnosis and benchmarking, including:

- Frame budget, requested frames, remaining frames, tubelet duplicates, and padding.
- Scout and rescue timestamps, observations, calibration, and effective temporal units.
- State runs, transition brackets, raw episode pairs, invalid-pair rejections, ranked candidates, and admission rejections.
- Refinement observations, final boundary corridors, and span provenance.
- Primary versus fallback source, fallback reason, Qwen call count, evidence packing, protected anchors, and role losses.
- Fallback support windows, occurrence cap, original fallback spans, dropped or clipped spans, and final constrained spans.

This makes it possible to distinguish whether a missed event was lost during scouting, rescue, pairing, candidate ranking, budget admission, boundary eligibility, or fallback filtering.
