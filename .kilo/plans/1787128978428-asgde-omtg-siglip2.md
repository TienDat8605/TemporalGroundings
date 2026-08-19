# ASGDE-OMTG With SigLIP2 Scout

## Goal

Add a new training-free `asgde-omtg` method for OMTG. It uses frozen SigLIP2 for full-video proposal retrieval and frozen `qwen3-vl-4b` for one absolute-source-time, multi-span grounding call. Preserve the existing `sgde-64` method unchanged.

## Fixed Decisions

- Register a new benchmark-specific method named `asgde-omtg`; do not mutate `sgde-64` semantics or its QVHighlights configuration.
- Pin the ASGDE scout to `google/siglip2-base-patch16-224`; do not change the project-wide `scout_features.DEFAULT_MODEL`.
- Use a 1 FPS full-video scout timeline with the existing robust normalization and conservative smoothing.
- Use Qwen3-VL-4B-Instruct as the initial frozen OMTG grounder, with dense evidence and no Mage or SemVID pruning in this experiment.
- Make exactly one `encode()` and one `predict()` call per query.
- Use a strict grounding-source-frame budget of 64 for zero/one confident separated peak and 128 for two or more confident separated peaks. A flat/uncertain timeline fails open to 64 uniform full-video frames; it never exceeds its selected budget.
- For multi-peak routes, select at most four separated candidate corridors after overlap/nearby-corridor merging and temporal diversity selection.
- Use 12 global anchors in the 64-frame route and 16 global anchors in the 128-frame route. Allocate all remaining frames to selected corridors. Deduplicate timestamps and deterministically refill from the eligible evidence regions so the final plan is exactly the selected hard budget.
- Pass `GroundingContext(0, sample.duration)` for every ASGDE prediction. All evidence timestamps and requested output spans are absolute source-video seconds, even when evidence is sampled from discontinuous corridors.

## Implementation Steps

1. Add `src/hybrid_vtg/methods/asgde_omtg/` with a new `ASGDEOMTG` `Method` implementation and OMTG-focused planning helpers.
   - Validate `sample.group == "omtg"` and require the encoded-evidence capability.
   - Reject encoder and post-encoder pruning, matching the current dense SGDE experimental contract.
   - Instantiate `ScoutProvider` with the SigLIP2 model ID, its pinned revision, 1 FPS, and runner-provided feature roots.
   - In `prepare()`, batch materialize the SigLIP2 timelines before loading Qwen and release the scout afterward.
   - In `run()`, build a timeline, extract proposals, choose the 64/128 route, build one chronological evidence plan, encode once, assign roles, predict once with full-video context, and return the Qwen multi-span result without router-based post-selection.

2. Make proposal extraction suitable for set-valued routing while retaining reusable SGDE primitives where behavior remains correct.
   - Include hysteresis components, penalized intervals, and multiscale density proposals in the candidate pool; the current SGDE extractor omits its implemented penalized-interval path.
   - Score deterministically, suppress strongly overlapping proposals, then merge overlapping or context-adjacent proposals into continuous candidate corridors before final temporal-diversity selection.
   - Define a confident separated peak as a retained candidate satisfying the predeclared score/contrast thresholds and not merged into another corridor. Select 128 frames only when there are at least two such corridors; otherwise select 64.
   - Retain at most four corridors for the 128-frame route. Rank by proposal score, then temporal diversity and timestamp tie-breakers. Do not use labels or OMTG test-set statistics to tune these constants.
   - Record all raw proposals, merged corridors, accepted/rejected candidates, peak count, and route reason in telemetry.

3. Implement deterministic multi-corridor evidence allocation in `asgde_omtg` planning code.
   - Build global anchors over the entire source duration first, excluding exact duplicates with local samples.
   - Pad every selected corridor on both sides using a declared adaptive halo derived only from corridor duration and uncertainty, clipped to video bounds.
   - Allocate each corridor a minimum before/onset/interior/offset/after sample pattern. Distribute residual local frames by normalized candidate score and uncertainty, using deterministic largest-remainder tie-breaking.
   - Sample locally in each padded corridor and label observations as `global_anchor`, `pre_context`, `onset_transition`, `candidate_interior`, `offset_transition`, or `post_context`; retain the originating corridor ID.
   - Sort chronologically, de-duplicate timestamps, and refill in a deterministic priority order to guarantee exactly 64 or 128 source frames. Fall-open routes contain only uniformly distributed `exploration` observations.
   - Keep candidate corridors separate in metadata. Do not collapse distant OMTG occurrences into one local grounding window.

4. Make SigLIP2 scout caching model-safe.
   - Update `ScoutProvider` feature discovery/loading so a provider configured for SigLIP2 only considers directories and artifacts explicitly identified as the SigLIP2 slug; it must not silently consume another scout's embeddings.
   - Add cache metadata for model ID, revision, FPS, embedding dimension, preprocessing/schema version, and video identity/revision. Reject mismatched or legacy artifacts and recompute them rather than mixing image/text encoders.
   - Ensure the visual and query embeddings used for a timeline have matching provenance. Emit the validated provenance and cache-hit status in ASGDE telemetry and manifests.
   - Keep this validation compatible with existing callers by applying it to newly written artifacts; update the feature extraction path as needed so SigLIP2 assets can be regenerated with the required metadata.

5. Update Qwen evidence prompting for sparse global ASGDE packs without changing span semantics.
   - For multi-span predictions with global context, explicitly say that frame labels are absolute source-video seconds, evidence is sparse and may contain temporal gaps, every visually supported occurrence must be returned separately in chronological JSON pairs, and unsupported intervals must be omitted.
   - Preserve `[]` as the absence response and preserve existing parsing/consolidation and duration clipping.
   - Do not use a local corridor offset for ASGDE; Qwen's existing relative-to-context conversion becomes identity because the context starts at zero.
   - Add a method-scoped prompt indicator or context metadata if required to avoid unintentionally changing prompts for the existing SGDE and other Qwen methods.

6. Wire the method into execution and reproducibility surfaces.
   - Register `asgde-omtg` in `src/hybrid_vtg/methods/__init__.py` and pass feature roots in `runner.py` for this method as well as `sgde-64`.
   - Add an `asgde-omtg` manifest configuration recording model/revision, scout FPS, smoothing/normalization and proposal constants, peak-to-budget decision, corridor cap, anchor counts, local allocation policy, strict budget scope, and one-call limits.
   - Add README and Idea 3 documentation that distinguish generic `sgde-64` from OMTG-specific ASGDE, state SigLIP2 is the ASGDE scout, identify Qwen3-VL-4B as the initial grounder, and avoid reusing QVHighlights results as OMTG evidence.
   - Document the intended invocation with `--benchmark omtg --model qwen3-vl-4b --method asgde-omtg`.

## Tests

1. Extend or add unit tests for proposal construction:
   - penalized proposals participate in the combined pool;
   - overlapping/nearby proposals merge while separated peaks remain distinct;
   - multi-peak selection caps at four with deterministic ordering;
   - flat or low-confidence timelines trigger a 64-frame full-video fallback.

2. Add evidence-planning tests:
   - one confident corridor yields exactly 64 frames, 12 anchors, chronological unique timestamps, and all before/interior/after roles;
   - two and four separated confident corridors yield exactly 128 frames, 16 anchors, representation from every admitted corridor, and no local-context collapse;
   - all route modes preserve one shared query-level budget, including duplicate/edge-of-video handling.

3. Add method integration tests with mock scout and mock Qwen backend:
   - OMTG multi samples make one encode and one predict call;
   - prediction context is `[0, duration]`;
   - multiple returned spans pass through chronologically;
   - non-OMTG samples and pruning options fail clearly;
   - telemetry records scout identity, cache provenance, candidate/corridor decisions, budget, roles, and call counts.

4. Add scout-cache tests:
   - a SigLIP2 provider cannot load a Nemotron or Qwen-embedding artifact;
   - stale revision/FPS/schema metadata is rejected and regenerated;
   - matching SigLIP2 video/query artifacts are reused.

5. Run focused tests first, then the full suite:
   - `pytest tests/test_sgde.py` plus new ASGDE tests;
   - `pytest`.

## Evaluation Plan

1. Run a small deterministic OMTG smoke subset with cache cold and warm runs. Verify exact 64/128 budgets, one grounder call, absolute output timestamps, no cache-model mismatch, and valid JSON multi-span outputs.
2. Run the full fixed 320-query OMTG split at seed 42 with Qwen3-VL-4B and report C-Acc, cardinality error, EtF1, tIoU, tP/tR/tF1 at 0.3/0.5/0.7, plus video-clustered paired bootstrap comparisons.
3. Report proposal diagnostics separately from final grounding: top-K target coverage, full target containment, endpoint availability, coverage stratified by target count/duration/dispersion, peak-count route distribution, and candidate admission losses.
4. Compare against the existing matched Qwen OMTG controls: uniform-one-shot-64f, embedding-window-local-64f, and a 128-frame full-video uniform control. Also run SigLIP2 ASGDE 64-only as an ablation to isolate the value of adaptive multi-peak allocation.
5. Report cold indexing time separately from cached-query scout time, plus source frames, decoded pixels, dense visual tokens, LLM input/output tokens, encode/predict latency, peak memory, and fallback rate. Do not claim a speedup without these matched measurements.

## Risks And Scope

- SigLIP2 was previously weak on repetitive TACoS actions; this OMTG adaptation is a benchmark-specific test, not evidence that SigLIP2 is generally sufficient. The full-video fallback and candidate-coverage metrics are mandatory safeguards.
- Four corridors cannot enumerate arbitrarily dispersed OMTG labels. The global anchors preserve global contrast, and route/candidate coverage must expose the resulting recall ceiling rather than hiding it.
- This change does not add boundary-refinement passes, Qwen embedding routing, learned calibration, fine-tuning, or spatial pruning. Those require separate matched-budget ablations after ASGDE routing is validated.
