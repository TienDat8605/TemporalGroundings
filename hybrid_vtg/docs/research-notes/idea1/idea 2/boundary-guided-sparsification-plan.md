# Boundary-Guided Sparsification Plan

## Goal

Add a training-free `boundary-guided-sparsification` (BGS) method that augments the existing frozen Qwen and UniVTG grounders. It should improve multi-interval OMTG grounding IoU by concentrating temporal detail near paired query-presence transitions, not by training a boundary model or replacing final grounding.

Add a matched `uniform-budget` baseline so BGS is compared at the identical duration-scaled sampled-frame budget.

## Requested Research Notes

The user requested the implementation plan and proposal under `hybrid_vtg/docs/research-notes/idea2/`. Planning-mode workspace permissions currently allow edits only in `.kilo/plans/`, so the requested proposal is temporarily stored at `.kilo/plans/1786588657038-boundary-guided-sparsification-proposal.md`.

When documentation editing is available, move this plan to `hybrid_vtg/docs/research-notes/idea2/implementation_plan_boundary_guided_sparsification.md` and move the proposal to `hybrid_vtg/docs/research-notes/idea2/proposal_boundary_guided_sparsification.md`. Do not alter their content during the move.

## Locked Decisions

- Primary evaluation: OMTG multi-interval grounding; retain compatibility with TACoS and both model backends. Qwen is the primary path; UniVTG is contract-tested.
- Coarse coverage is one frozen observation per fixed 8-second cell. A cell asks whether it supports the query, not whether generic motion occurs.
- Per-video temporal presence is the top-quartile mean of Qwen's spatial-row query scores at a timestamp. UniVTG's one row per timestamp is used directly.
- Normalize each query-video timeline with median/MAD. States are `absent` for `z <= 0`, `present` for `z >= 1`, and `uncertain` otherwise. Require two adjacent coarse observations to confirm a state change.
- A detected `absent -> present` bracket is a start candidate; `present -> absent` is an end candidate. Pair chronologically, retain at most six scored episode pairs, and support all retained pairs for multi-cardinality samples.
- A boundary is refined only locally. Its maximum depth is derived as `ceil(log2(bracket_width / 1 second))`; for an 8-second bracket this is three rounds. An uncertain midpoint gets one pair of quarter-point probes. Persistent uncertainty remains a wider boundary corridor rather than a forced timestamp.
- Total sampled-frame budget per sample is `B(D) = max(64, even(ceil(D / 8) + 64))`, where `D` is video duration. The even rounding is required for Qwen temporal tubelets. Scout observations, refinement probes, and supplementary final evidence must all fit this one ledger. The uniform baseline uses the same function.
- Reuse HMVE's `pack_evidence` and its 12.5% final post-encoder retention default. Protect global anchors and selected start/end evidence from being dropped. Do not add a new spatial allocator.
- This is a frozen, deterministic inference policy. Do not add trainable heads, learned selectors, benchmark-fitted thresholds, or a new final prediction interface.

## Reuse Map

| Existing code | Reuse in BGS |
|---|---|
| `contracts.py` (`TemporalEvidence`, `ModelBackend`, `GroundingContext`) | No contract changes; BGS remains backend-agnostic. |
| `media.py` (`uniform_timestamps`) | Uniform scout and matched baseline timestamps. |
| `methods/hmve/__init__.py` (`pack_evidence`, `observation_timestamps` patterns) | Final compact packing, role propagation, Qwen even-frame handling patterns, and protected anchors. |
| `models/qwen.py` | Existing `encode`, `query_scores`, timestamp-role prompt, and one `predict` call. |
| `models/univtg/__init__.py` | Existing sparse absolute timestamp support and evidence-unit cap. |
| `metrics.py` and OMTG adapter | Existing multi-span metrics; no evaluation semantic changes. |

## Implementation Tasks

1. Add shared budget helpers in a small new method-local utility module or the new BGS module.
   - Implement `duration_budget(duration)`: compute the locked `B(D)` and return an even total.
   - Implement `scout_timestamps(duration)`: one center timestamp per 8-second cell, padded by repeating the final real timestamp only when Qwen requires an even batch.
   - Track `requested_frames`, `duplicate_padding_frames`, and `remaining_frames`. Count padding against the ledger.
   - Fail with an assertion if any route requests more than `B(D)` sampled frames.

2. Add `src/hybrid_vtg/methods/uniform_budget/__init__.py`.
   - Register name: `uniform-budget`.
   - Sample exactly `B(D)` uniform timestamps over the full video; the even budget avoids artificial padding.
   - Encode once, score once, assign `global_anchor` roles, and compact with reused `pack_evidence` at 12.5% retention while respecting `model.maximum_evidence_units`.
   - Call `model.predict` exactly once with full-video `GroundingContext`.
   - Emit telemetry: budget, sampled frames, encoder/query-score/prediction calls, created/retained evidence, retention ratio, and `policy: uniform`.

3. Add `src/hybrid_vtg/methods/boundary_guided_sparsification/__init__.py`.
   - Define focused immutable structures for coarse temporal observations, boundary brackets, refined boundary corridors, and candidate episode pairs.
   - Keep constants declared at module level: 8-second cells, two-cell persistence, robust state thresholds, maximum six pairs, one-second target resolution, one ambiguity-probe round, 12.5% retention.
   - Do not expose a large new CLI configuration surface for the MVP; record every constant in telemetry.

4. Implement robust query-presence aggregation.
   - Group `TemporalEvidence` rows by timestamp.
   - For a group with one row, retain that score. For Qwen-style multi-row groups, sort scores and average the upper quartile using a deterministic ceiling count of at least one row.
   - Robustly normalize the ordered timestamp scores with median and MAD. Handle zero MAD deterministically: if all scores are equal, mark all observations uncertain rather than manufacturing a transition.
   - Add unit-level metadata for raw, aggregated, normalized score, and assigned state.

5. Implement coarse episode and pairing logic.
   - Convert the normalized sequence into persistent absent/present runs; uncertain observations neither establish nor erase a state by themselves.
   - Create start/end brackets only when a confirmed state transition has known observations on both sides. Preserve video-edge episodes as open-ended candidates with a boundary role at `0` or `duration` rather than silently dropping them.
   - Pair starts to the next compatible end in chronological order. Score pairs with start/end transition confidence plus mean inside presence minus immediate outside presence.
   - For `Sample.cardinality == "multi"`, keep up to six non-overlapping highest-scoring pairs. For single-cardinality samples, retain the top pair for local refinement, while preserving global anchors for final grounding.
   - If no confident pair exists, do not invent one: use scout evidence and a deterministic full-timeline anchor fallback under the same budget.

6. Implement batched adaptive boundary refinement.
   - Refine all active start/end brackets in batches by depth, rather than making one model call per boundary.
   - At each midpoint, derive the new state from the same robust scoring policy. If the midpoint is confidently one of the endpoint states, retain the half-bracket that preserves the directional state change.
   - If uncertain, issue at most one batched quarter-point probe pair for that boundary. Resolve it only when the probes form a directional absent/present or present/absent contrast.
   - Stop at one-second width, derived depth exhaustion, successful resolution, or persistent ambiguity. Record the final uncertainty interval and reason.
   - Before every batch, trim lower-ranked candidate pairs so that the batch plus all already-spent frames and reserved budget fit `B(D)`. Never exceed the ledger.

7. Build the final role-aware evidence pack and call the unchanged grounder once.
   - Concatenate the coarse evidence and all refinement evidence with `TemporalEvidence.concatenate`.
   - Assign every row one role: `global_anchor`, `interior`, `start`, `end`, or `ambiguous_boundary`. A row inside more than one region uses deterministic priority: start/end, ambiguous boundary, interior, global anchor.
   - Reuse `pack_evidence`; construct protected anchor indices from temporal global coverage and at least one highest-scoring row for each selected boundary corridor. Limit the compact target by `model.maximum_evidence_units` where present.
   - Make exactly one `model.predict(sample, compact, GroundingContext(0, duration))` call. Do not use coarse retrieval scores to rerank or replace its predicted spans.
   - Emit complete telemetry: budget ledger; coarse cells; raw/aggregated/normalized scores; state runs; brackets; probes; resolved/unresolved corridors; candidate pairs; selected roles; all call counts; created and retained evidence; and whether fallback was used.

8. Register both methods and update public method documentation.
   - Update `src/hybrid_vtg/methods/__init__.py` to register `BoundaryGuidedSparsification` and `UniformBudget` alongside the existing methods.
   - Update the README method overview and “only three methods” language to describe the two additional methods accurately, including that `uniform-budget` is a matched experimental baseline.

## Test Plan

1. Extend `tests/test_methods.py` with pure helper tests.
   - `B(D)` is even, at least 64, and equals the approved schedule for short, average-duration, and 17-minute videos.
   - Scout timestamp count uses one observation per 8-second cell and counts Qwen padding.
   - Top-quartile aggregation rejects a single-row outlier compared with maximum aggregation.
   - Median/MAD state assignment covers absent, present, uncertain, and zero-MAD timelines.
   - Persistent transitions generate the correct start/end brackets, retain edge episodes, and do not make transitions from isolated noise.
   - Chronological pairing produces the intended pairs for repeated occurrences and never crosses pairs.
   - Three refinement rounds reduce an 8-second directional bracket to at most one second.
   - Ambiguous midpoint behavior performs at most one two-sided probe and returns an unresolved corridor rather than a fabricated point.

2. Add fake-backend integration tests.
   - Implement a deterministic `ModelBackend` test double that returns timestamp-aligned evidence and scripted query scores.
   - Verify BGS works with one evidence row per timestamp and Qwen-like multiple rows per timestamp.
   - Verify single-cardinality and multi-cardinality paths produce exactly one final prediction call.
   - Verify all requested scout/probe/supplementary frames are within `B(D)` and telemetry matches the ledger.
   - Verify compact evidence respects a bounded backend's `maximum_evidence_units`.
   - Verify `uniform-budget` samples exactly `B(D)` frames and performs one encode/predict route.

3. Run static and existing regression checks.
   - `ruff check src tests`
   - `pytest`

4. Run the initial OMTG experiment after unit validation.
   - Compare `uniform-budget` and `boundary-guided-sparsification` with Qwen as primary under the same `B(D)` schedule and 12.5% packing policy.
   - Report existing OMTG multi-span metrics plus method telemetry: target coverage and endpoint availability as diagnostic-only evaluation fields, number of selected pairs, unresolved-boundary rate, fallback rate, sampled-frame budget, encoder calls, retained evidence, and wall time.
   - Do not claim an efficiency gain unless cold-query timing includes scout and refinement work. Compare cached-index timing only as a separately labelled result.
   - Contract-test the new method with UniVTG before making Qwen-only benchmark claims; assess TACoS only after OMTG behavior is stable.

## Acceptance Conditions

- The new method remains fully training-free and uses one final grounder prediction call per sample.
- It preserves the exact duration-scaled sampled-frame ledger and the matched baseline uses the same ledger.
- It does not force ambiguous boundaries into false precision.
- It returns multi-occurrence evidence to the final grounder without cross-pairing boundaries.
- It preserves Qwen and UniVTG compatibility through the existing `ModelBackend` interface.
- Any IoU claim is made only against `uniform-budget` under matching `B(D)`, model, packing ratio, and test subset.

## Risks and Guardrails

- Eight-second single-view scouting can miss events entirely between observations. This is an explicit recall limitation; record it through oracle coverage diagnostics rather than concealing it with post-hoc routing.
- The duration-scaled budget grows for very long videos. Uniform baselines must use the same budget; report per-duration strata and never compare it directly to a fixed-64 baseline as an efficiency win.
- Qwen query scores are patch-derived and not calibrated presence probabilities. Robust aggregation, MAD normalization, persistence, and ambiguity retention are safeguards, not guarantees.
- Existing HMVE packing may prioritize semantic peaks over some protected roles. Enforce the protected anchor floor before filling remaining capacity; log any cap-induced role losses.
