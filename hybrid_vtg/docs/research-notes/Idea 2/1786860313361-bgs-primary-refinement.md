# BGS Primary Refinement Plan

## Goal

Restore the intended boundary-guided coarse-to-fine policy for Qwen:

```text
8-second query-presence scout
-> directional absent/present brackets
-> budgeted binary boundary refinement
-> resolved BGS episode pairs are the returned predictions
```

Qwen generation remains a fallback only when BGS produces no fully resolved interval.

## Locked Decisions

- Preserve one logical 8-second scout observation per Qwen temporal tubelet.
- Reserve 64 *physical Qwen frames* after the scout for boundary refinement.
- Keep paired boundary refinement directional: absent-to-present for starts and present-to-absent for ends.
- Select candidate pairs before refinement using an upper-bound frame cost; do not select 16 then silently discard pairs during refinement.
- Emit a BGS span only when both its start and end corridors resolve to the one-second target. Omit pairs with an unresolved boundary rather than fabricating a point estimate.
- Use the existing Qwen full-timeline generation only if BGS has zero resolved spans, including when no pair is detected, no pair fits the budget, or all selected pairs remain ambiguous.
- Do not allow a Qwen fallback result to be mixed with resolved BGS spans.
- Keep the existing one final-prediction-call maximum. Successful primary BGS predictions make zero `model.predict()` calls; fallback makes exactly one.

## Implementation Steps

1. Update `src/hybrid_vtg/methods/budget.py` to expose the Qwen-aware BGS budget schedule.
   - Compute `coarse_cells = ceil(duration / CELL_SECONDS)`.
   - Change `duration_budget()` to return `even(max(64, 2 * coarse_cells + 64))`.
   - Add a named `REFINEMENT_RESERVE_FRAMES = 64` constant rather than embedding the value in the formula.
   - Update the docstring to state that `2 * coarse_cells` pays for tubelet-preserving scouting and the reserve pays for local refinement.
   - Continue accounting for every requested physical frame through `BudgetLedger`.

2. Keep `uniform-budget` matched to the revised physical-frame budget.
   - It should continue sampling exactly `duration_budget(sample.duration)` timestamps and calling Qwen once.
   - Add telemetry identifying the matched budget schedule and configured refinement reserve, so reports make the comparison explicit.

3. Replace fixed maximum-pair selection with refinement-cost admission in `boundary_guided_sparsification`.
   - Restore a small hard safety cap (`MAXIMUM_PAIRS = 6`), but admit only the highest-scoring non-overlapping candidates whose estimated worst-case work fits the 64-frame reserve.
   - Estimate each active non-edge boundary as:
     ```text
     midpoint cost = depth * 2 physical frames
     ambiguity allowance = 4 physical frames
     ```
     where `depth = ceil(log2(bracket_width / TARGET_RESOLUTION))` and the extra four frames are the one logical quarter-probe pair encoded as a Qwen tubelet pair each.
   - Sum both boundaries for each pair. Edge boundaries cost zero.
   - Derive the usable reserve from `ledger.remaining_frames` immediately after scouting, not from a hard-coded assumption.
   - Select candidates in current score order only while the full estimated cost fits. Record candidates rejected for budget in telemetry.
   - Remove or narrow `_fit_batch()` so it is a final assertion/defensive guard, not the normal pair-selection mechanism. It must never cause a pair that began refinement to disappear silently.

4. Retain the existing batched directional binary-search mechanics, with explicit per-pair refinement state.
   - For a start bracket, retain the half bracket consistent with `absent -> present`.
   - For an end bracket, retain the half bracket consistent with `present -> absent`.
   - Preserve the fixed scout calibration during local observations; do not recalibrate midpoint/probe scores from a tiny local batch.
   - On an uncertain midpoint, perform at most one directional quarter-probe round.
   - Mark a corridor `resolved` only at `TARGET_RESOLUTION`; leave persistent uncertainty/depth exhaustion as `ambiguous`.
   - Return per-boundary final left/right witnesses in refinement telemetry so endpoint decisions are auditable.

5. Build primary spans from resolved refined pairs before preparing the final evidence pack.
   - Index final corridors by `(pair_id, role)` after refinement and removal of budget-rejected pairs.
   - A pair is output-eligible only when it has both a resolved `start` and resolved `end` corridor and the resulting end is after the start.
   - Convert each resolved corridor to its midpoint endpoint. The maximum nominal endpoint error is half of `TARGET_RESOLUTION`.
   - Construct chronological `ScoredSpan` values from eligible pairs using a deterministic confidence derived from pair and boundary confidence.
   - Omit every pair with an ambiguous/missing start or end corridor; record each omission and reason in telemetry.
   - Serialize the BGS spans into `raw_output` in the same JSON-pair format as the backend for downstream inspection.

6. Make BGS spans the primary `Prediction` result.
   - If one or more resolved BGS spans exist, return them directly and do not call `model.predict()`.
   - Set telemetry fields such as:
     ```text
     prediction_source: "bgs-primary"
     llm_or_fusion_calls: 0
     resolved_pair_ids: [...]
     omitted_pair_ids: [...]
     qwen_fallback_used: false
     ```
   - Still create and pack evidence only if needed for retained diagnostics; prefer skipping merged-score/packing work on the primary path unless its telemetry is an established reporting requirement.
   - If no resolved BGS span exists, preserve the current fallback-anchor/evidence path and call Qwen once. Mark:
     ```text
     prediction_source: "qwen-fallback"
     qwen_fallback_used: true
     fallback_reason: "no_candidate" | "budget_rejected" | "unresolved_boundaries"
     llm_or_fusion_calls: 1
     ```
   - Never merge fallback Qwen spans with partial BGS spans.

7. Change final evidence protection for fallback generation.
   - When fallback packing is used, protect both temporal witnesses for each boundary corridor: one on the left state side and one on the right state side.
   - Retain the existing global timeline coverage anchors.
   - Replace the current one-highest-score-per-corridor protection rule, which can discard the absent/present contrast required to interpret a transition.

8. Update unit and integration tests in `tests/test_methods.py`.
   - Update budget expectations for representative short, medium, and long durations. Assert `duration_budget(D) >= 2 * ceil(D / 8) + 64`, subject to even rounding.
   - Assert BGS has exactly 64 frames remaining immediately after a Qwen scout on durations where the minimum budget does not dominate.
   - Add pair-cost admission tests: selected pair costs fit the available reserve; rejected pairs are reported; no selected pair is dropped later by batching.
   - Retain directional binary-search tests for start and end boundaries, and add a Qwen-tubelet version that verifies the physical-frame accounting.
   - Test primary output: resolved start/end corridors produce chronological BGS spans and no `predict()` call.
   - Test ambiguous output: a pair with either ambiguous corridor is omitted; if all pairs are omitted, exactly one Qwen fallback `predict()` call occurs.
   - Test fallback output: no candidate pair produces one Qwen prediction and correctly labeled telemetry.
   - Test that a successful BGS-primary result has `llm_or_fusion_calls == 0`, while a fallback has `== 1`.
   - Update existing expectations that currently encode `MAXIMUM_PAIRS = 16`, `duration_budget(17 * 60) == 256`, and one final prediction on all BGS paths.

## Evaluation Protocol

1. Rerun the same 32-sample OMTG seed with `--rerun` for both `boundary-guided-sparsification` and `uniform-budget`.
2. Inspect `errors.jsonl` separately; failures must not be conflated with explicit empty predictions or BGS omissions.
3. Report, alongside existing OMTG metrics:
   - primary-BGS rate versus Qwen-fallback rate;
   - scout candidate-pair coverage of targets;
   - resolved-pair coverage and endpoint error before fallback generation;
   - budget-rejected pair count;
   - ambiguous-boundary omission rate;
   - requested frames, refinement frames, and remaining frames;
   - exact-cardinality accuracy for BGS-primary outputs and fallback outputs separately.
4. Compare BGS only against the revised matched-budget `uniform-budget` run. Do not compare results from the current zero-refinement budget schedule.

## Risks and Guardrails

- A 64-frame reserve supports only a few complete Qwen boundary searches. Budget admission must favor full refinement of the highest-quality pairs over broad but incomplete coverage.
- Returning only fully resolved pairs improves precision and makes BGS structurally primary, but can reduce recall on intermittent or short events.
- The 8-second/two-cell scout still cannot reliably propose very short occurrences. This is out of scope for this change and should be reported by the scout-coverage diagnostic.
- The revised budget is intentionally larger than the current schedule. Matching `uniform-budget` is required for a fair accuracy comparison.
- Skipping final Qwen generation on the BGS-primary path changes the method from “BGS evidence for Qwen” to “BGS temporal grounder with Qwen fallback”; telemetry and experiment labels must state this clearly.
