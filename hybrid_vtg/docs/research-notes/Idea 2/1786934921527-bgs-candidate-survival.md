# BGS Candidate Survival and Fallback Constraint Plan

## Goal

Address the five observed BGS weaknesses without changing the duration-matched total-frame budget:

1. retain useful ambiguous boundary candidates as conservative BGS predictions;
2. rank/admit episodes by reliability and resolvability rather than contrast alone;
3. prevent distant or internally inconsistent boundary pairs;
4. use part of the fixed refinement reserve for a denser adaptive rescue scout;
5. constrain the remaining Qwen fallback to BGS-supported candidate windows and count.

## Locked Decisions

- Keep `duration_budget()` and `UniformBudget` unchanged. Reallocate within BGS's existing 64 physical post-scout frames.
- Reserve 16 physical frames for the rescue scout (`8` logical observations for Qwen tubelets; `16` for non-tubelet backends). The remaining ledger capacity pays for refinement. Do not reserve frames that are not used.
- If at least one selected pair has chronological endpoints, return BGS-primary spans even when either corridor is ambiguous. Use corridor midpoints; penalize ambiguous/wide corridors in the `ScoredSpan.score`. Do not invoke Qwen fallback in this case.
- Continue to omit invalid pairs: missing corridor, non-chronological endpoints, or a corridor whose width exceeds the configured conservative maximum.
- Retain the current 1-second resolution goal. Define a bounded conservative ambiguity maximum (8 seconds per boundary) so a broad uncertain corridor cannot create an uncontrolled span.
- Qwen fallback is only for no viable BGS spans. Pass BGS candidate windows and an expected maximum occurrence count in `GroundingContext`; Qwen receives explicit prompt guidance. BGS also deterministically clips/filters fallback spans to padded candidate windows and caps them at the supported candidate count.
- The candidate metadata in `GroundingContext` is optional/default-empty so existing callers and non-generative backends continue to work unchanged.

## Implementation

1. Extend `src/hybrid_vtg/contracts.py`.
   - Add immutable optional fields to `GroundingContext`:
     - `candidate_windows: tuple[tuple[float, float], ...] = ()`
     - `maximum_occurrences: int | None = None`
   - Validate neither here nor in backends; BGS owns construction and clipping. Existing context construction remains valid.

2. Add BGS constants and temporal helpers in `src/hybrid_vtg/methods/boundary_guided_sparsification/__init__.py`.
   - Add named constants for the 16 physical rescue-frame allocation, rescue subdivision (two observations per chosen 8-second cell), candidate-window padding, maximum conservative boundary width, ambiguity confidence penalty, interval-length penalty, and interior-consistency penalty.
   - Add helpers that:
     - rank rescue cells from the fixed scout calibration, prioritizing isolated present cells and high-scoring uncertain cells, then deterministic timestamp order;
     - produce two 4-second-subcell timestamps for each admitted rescue cell, deduplicated and clipped to video duration;
     - merge scout and rescue observations in chronological order while preserving their common global calibration;
     - calculate pair reliability from directional boundary confidence, witness state strength, interval compactness, and the proportion of confident-present interior observations;
     - identify an invalid/distant pair when its interval exceeds the configured maximum derived from cell scale or contains a sufficiently strong absent interior run;
     - create conservative endpoints from corridor midpoints and calculate a score penalty by ambiguity status and corridor widths;
     - construct padded candidate windows, clip spans to their best-overlap window, and rank/cap fallback spans deterministically.

3. Run the rescue scout before candidate detection in `BoundaryGuidedSparsification.run()`.
   - After the fixed 8-second scout and its calibration, allocate up to 16 physical frames from `ledger.remaining_frames` to rescue timestamps.
   - Use the existing `observe(..., preserve_logical_observations=True)` path so Qwen tubelet accounting and global score calibration are preserved.
   - Label rescue evidence with a dedicated `rescue_scout` role/stage and merge its observations with the coarse timeline before calling `detect_boundaries()` and `pair_boundaries()`.
   - Record rescue candidates, requested logical/physical timestamps, observations, frames consumed, and effective temporal units in telemetry.
   - Leave unused rescue allocation available to refinement; all later admission must use actual `ledger.remaining_frames`.

4. Replace contrast-only episode ranking and permissive pairing.
   - Update `pair_boundaries()` to reject pairs that fail the distant-pair/interior-absence checks and return explicit rejection diagnostics alongside valid candidates, or add a dedicated filtering step immediately after pairing with identical telemetry.
   - Replace the current `start.confidence + end.confidence + contrast` score with the documented reliability score. Preserve deterministic score/timestamp/pair-id ties.
   - Ensure `select_refinement_pairs()` uses that reliability score, keeps non-overlap and cardinality behavior, and continues to reject only when complete worst-case refinement cannot fit the remaining frame ledger.
   - Emit telemetry separately for raw transition pairs, invalid-pair rejections, ranked candidate pairs, selected pairs, and budget/cap/overlap rejections. This makes a near-target candidate's loss attributable to pairing, ranking, admission, or refinement.

5. Make ambiguous corridors output-eligible conservatively.
   - Replace `_resolved_spans()` with an eligibility builder that accepts both resolved and bounded ambiguous corridors.
   - A pair is eligible when both corridors exist, each is resolved or has width at most the conservative maximum, and midpoint endpoints are chronological.
   - For an eligible ambiguous boundary, retain the midpoint and apply the configured ambiguity/width penalty to the span score. Mark each span's provenance (`resolved`, `conservative-start`, `conservative-end`, or `conservative-both`) in telemetry.
   - Keep fully unresolved/distant pairs omitted with explicit reasons. Rename/add telemetry so resolved and conservative pair ids are reported separately.
   - Return `prediction_source: "bgs-primary"` whenever this builder returns at least one span, with no `model.predict()` invocation. `raw_output` remains JSON span pairs.

6. Constrain the no-span fallback path.
   - Build fallback support windows from the top-ranked non-overlapping valid candidates, preferring selected/refined candidates and falling back to ranked candidates when none were selected. Pad and merge overlapping support windows; use their count as the occurrence cap.
   - Pass those windows and cap through the extended `GroundingContext` while retaining global `start=0`, `end=sample.duration`.
   - Update `QwenEvidenceBackend._prompt()` to state that multi-occurrence output must be confined to the supplied candidate windows and must not exceed the supplied maximum; include the windows in source-video seconds. Preserve the existing prompt when context has no constraints.
   - After `model.predict()`, BGS clips each returned span to its best-overlap support window, drops zero-overlap spans, applies temporal NMS, and retains at most the candidate-window count in deterministic model-score order. If no support windows exist, do not invent a cap/filter and preserve current fallback behavior.
   - Record original fallback spans, support windows, cap, spans dropped/clipped, and final constrained spans in telemetry.

7. Update tests in `hybrid_vtg/tests/test_methods.py`.
   - Update the ambiguous-corridor test: bounded ambiguous corridors produce BGS-primary conservative spans, make zero `predict()` calls, and report conservative provenance.
   - Add an over-wide ambiguous corridor case that remains ineligible and falls back.
   - Add rescue-scout tests for Qwen tubelets: rescue uses at most 16 physical frames, preserves logical grouping/calibration, adds a candidate missed by the fixed scout, and leaves correct remaining budget for refinement.
   - Add pairing/ranking tests: reject a distant pair with an absent interior run; rank a compact two-sided reliable pair over a high-contrast but unreliable pair; retain deterministic ties.
   - Add fallback-context tests with a scripted backend: candidate windows/count are delivered, output is clipped and capped, zero-overlap spans are removed, and unconstrained no-candidate fallback preserves legacy output behavior.
   - Preserve and extend ledger assertions: requested frames never exceed `duration_budget`, every tubelet duplicate is accounted for, and admission uses remaining frames after rescue.
   - Update expected telemetry/source assertions affected by conservative primary output.

8. Validate.
   - Run `pytest hybrid_vtg/tests/test_methods.py` and the full test suite.
   - Rerun the identical OMTG 32-sample seed for BGS and matched `uniform-budget` with `--rerun`.
   - Report global metrics plus BGS-primary resolved/conservative/fallback counts; candidate coverage at IoU 0.3 and 0.5; candidate loss by invalid-pair, admission, and ambiguity reason; and fallback spans before/after support filtering.

## Risks and Guardrails

- Conservative ambiguous spans can increase false positives. The maximum corridor width, explicit score penalty, and maximum six admitted pairs limit this.
- Reserving rescue frames can reduce refinement capacity. The rescue allocation is capped and any unused frames remain available; tests must cover admission after rescue.
- Candidate-window filtering may suppress a true fallback span when the scout misses it. Apply hard filtering only when at least one valid BGS support window exists; retain unconstrained fallback for no-candidate cases.
- Context constraints are advisory for non-Qwen backends; BGS's post-filter is the enforcement point and must remain backend-agnostic.
- Do not use benchmark targets in method logic. Ground truth is only used in post-run diagnostic aggregation.
