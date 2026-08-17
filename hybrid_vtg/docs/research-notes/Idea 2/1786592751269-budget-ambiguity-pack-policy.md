# Plan: Revert Misunderstood Change, Implement Useful Tubelet Micro-Windows

## Objective
Replace the misunderstood corridor-level `±0.25s` change with the intended frame-level change: keep Qwen 2-frame tubelets, but make each pair a useful micro-window instead of duplicated `[t, t]` timestamps.

## Confirmed Decisions
1. Keep combined prior scope items:
   - Budget: `max(64, 2 * ceil(duration / 8))`
   - BGS present threshold: `0.75` (absent remains `0.0`)
   - Default pack ratio: `0.25` in BGS and `uniform-budget`
   - Multi selected-pair cap: `16`
2. Keep coarse routing at 8s cells.
3. `±0.25s` applies to **all Qwen duplication calls** (scout + midpoint + ambiguity-probe + supplementary), not just scout.
4. Do **not** do corridor/post-eval expansion as the implementation mechanism.

## Revert Scope (misunderstood behavior)
Remove any logic/tests that implement `±0.25s` by expanding boundary corridors (e.g., zero-width `[t,t]` corridor expansion). The feature must be implemented at tubelet timestamp generation, not boundary formatting.

## Implementation Plan
1. Update `hybrid_vtg/src/hybrid_vtg/methods/budget.py`.
   - Keep `CELL_SECONDS = 8.0`.
   - Keep `duration_budget()` as `max(64, 2 * ceil(duration / CELL_SECONDS))` with positive-duration validation and even-budget invariant handling.
   - Replace/extend `duplicate_tubelets(...)` behavior for Qwen mode:
     - For each logical timestamp `t`, produce physical pair:
       - `left = max(0.0, t - 0.25)`
       - `right = min(duration, t + 0.25)`
     - Preserve roles as duplicated pair roles.
     - Return tubelet duplicate accounting unchanged (`+1` per logical timestamp).
   - Add an explicit logical mapping artifact (returned or encoded via metadata contract) so both physical timestamps map back to one logical observation index.

2. Update BGS to use logical-cell aggregation when Qwen tubelet mode is active in `hybrid_vtg/src/hybrid_vtg/methods/boundary_guided_sparsification/__init__.py`.
   - Ensure aggregation groups by logical tubelet index (pair id), not raw timestamp value, when duplication is active.
   - Preserve coarse unit count as `C = ceil(duration/8)` (not `2C`).
   - Apply same logical grouping for refinement observations created via duplication (midpoint/probe/supplement) so state transitions remain stable and deterministic.
   - Keep constants updates in this file:
     - `PRESENT_THRESHOLD = 0.75`
     - `RETENTION_RATIO = 0.25`
     - `MAXIMUM_PAIRS = 16`

3. Update `hybrid_vtg/src/hybrid_vtg/methods/uniform_budget/__init__.py`.
   - Set default `RETENTION_RATIO = 0.25`.
   - Keep using shared `duration_budget()`.

4. Keep boundary corridor logic unchanged except for removing misunderstood expansion path.
   - No special corridor widening policy is required for this feature.
   - `video_edge` and other corridor widths remain driven by existing refinement/detection semantics.

## Test Plan (`hybrid_vtg/tests/test_methods.py`)
1. Update budget expected values for `max(64, 2 * ceil(D/8))`.
2. Add/adjust state-threshold assertions for `0.75` present cutoff.
3. Add pair-cap test validating multi selection max is 16.
4. Replace corridor-expansion test with tubelet micro-window tests:
   - Qwen duplication produces per-logical pair `[t-0.25, t+0.25]` clamped to video bounds.
   - Duplicate accounting remains correct.
   - Logical aggregation still yields one coarse observation per 8s cell.
5. Keep existing ledger invariants:
   - requested frames == budget where expected
   - remaining frames accounting unchanged
   - Qwen duplication telemetry still consistent.

## Validation
1. Run `pytest tests/test_methods.py`.
2. Run full `pytest`.
3. Spot-check BGS telemetry on a small OMTG run:
   - `coarse_units == ceil(duration/8)`
   - scout physical frames reflect 2 per logical cell
   - constants show `present_threshold=0.75`, `maximum_pairs=16`, `retention_ratio=0.25`
   - fallback rate/empty-output diagnostics recorded for comparison against prior run.

## Risks / Guardrails
1. If aggregation remains timestamp-based after micro-windowing, coarse units will double and routing behavior will drift; logical-index grouping is mandatory.
2. Edge clamping can collapse micro-window width near video boundaries; this is acceptable and deterministic.
3. Combined knob changes (budget/threshold/pack/pair-cap + micro-window tubelets) should be compared with clear run labels to attribute effects.
