# Residual-search v2: aggregate result assessment

## Experiment plan

### Objective

Test whether a strictly training-free, budgeted residual search can improve
one-to-many temporal grounding by repeatedly selecting windows that may contain
uncovered occurrences, rather than selecting a fixed top-K once.

The intended contribution was an anytime inference wrapper around a frozen
TimeLens2 grounder:

\[
\text{route evidence}
\rightarrow
\text{ground one window}
\rightarrow
\text{update unexplained evidence}
\rightarrow
\text{stop when covered}.
\]

### Fixed policy

- Preserve the deterministic 20--60 second v1 windows and frozen embedding
  router/grounder.
- Evaluate hard budgets B=64 and B=128, counting router and localizer frames.
- Reserve at least eight frames for local grounding and reduce router frames
  per window only when necessary to remain within budget.
- Give every local action one fixed, even frame allocation for that
  video/query/budget.
- Rank unvisited windows using equal-weight router relevance, temporal novelty,
  and the fraction not explained by the current merged interval set.
- Stop after two consecutive calls add less than one second of support and the
  residual relevance mass is at most 0.25; also stop when candidates or budget
  are exhausted.
- Retain the v1 interval mapping and heuristic merger so the experiment isolates
  routing/allocation/stopping rather than changing the decoder.

### Controls

1. `score-window-local`: identical budget and stopping logic, but visit windows
   in router-score order. This isolates residual reranking.
2. `residual-window-local`: full residual ranking and adaptive stopping.
3. `residual-window-local-no-stop`: residual ranking without the stability
   stop. This isolates early stopping.

All three schedules use the same frozen TimeLens2-4B checkpoint, controlled
prompt, 320 OMTG queries, and deterministic decoding.

### Measurements and success criteria

Primary quality metrics are C-Acc, EtF1, tIoU, and CardinalityError. Diagnostic
and efficiency metrics are per-occurrence RouterRecall, same-call-count
OracleRouterRecall, total frames, budget overflow, model calls, synchronized
model time, wall time, visited-window fraction, early-exit rate, and stop
reasons.

The plan would be supported if residual search improved the quality/compute
frontier over v1 embedding-window search, residual ordering beat score ordering,
and early stopping saved meaningful compute without degrading the interval set.
Zero budget overflow and complete 320-query execution were engineering
requirements rather than evidence of algorithmic improvement.

## Verdict

The v2 implementation succeeds as an engineering result: all six runs cover
320 samples, respect the frame budget with zero overflow, and retain high
per-occurrence router recall. It does **not** improve the original v1
embedding-window method. At both budgets it is less accurate, uses more model
calls, and takes longer.

Residual reranking also provides almost no benefit over score ordering, while
the stopping rule is effectively inactive. The current v2 should therefore be
reported as a negative ablation, not as the main method.

This assessment uses aggregate summaries supplied on 2026-07-26. Raw v2
predictions were not available locally, so differences below do not yet have
paired video-clustered confidence intervals.

## Main v2 results

| Schedule | B | C-Acc ↑ | EtF1 ↑ | tIoU ↑ | Card. error ↓ | Frames | Calls | GPU s | Wall s | Router recall ↑ | Early exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Score order | 64 | 23.12 | 11.34 | 38.00 | 1.716 | 19,906 | 835 | 532.67 | 771.33 | 88.90 | 0.00% |
| Residual | 64 | 23.12 | 11.34 | 37.79 | 1.719 | 19,906 | 835 | 532.61 | 771.30 | 88.79 | 0.00% |
| Residual, no stop | 64 | 23.12 | 11.34 | 37.79 | 1.719 | 19,906 | 835 | 532.55 | 771.33 | 88.79 | 0.00% |
| Score order | 128 | 26.56 | 15.95 | 45.03 | 1.663 | 40,326 | 1,200 | 911.89 | 1,272.42 | 97.87 | 1.56% |
| Residual | 128 | 26.88 | 16.07 | 45.04 | 1.656 | 40,362 | 1,203 | 912.53 | 1,273.17 | 98.08 | 1.25% |
| Residual, no stop | 128 | 26.88 | 16.07 | 45.01 | 1.653 | 40,438 | 1,210 | 914.49 | 1,275.43 | 98.08 | 0.00% |

## Comparison with v1 embedding-window search

| Candidate | Δ C-Acc | Δ EtF1 | Δ tIoU | Δ card. error | Frame change | Call change | GPU-time ratio | Wall-time ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual B=64 vs v1 B=64 | -13.12 | -7.14 | -3.15 | +0.206 | -722 | +134 | 1.38× | 1.17× |
| Residual B=128 vs v1 B=128 | -6.25 | -5.07 | -1.54 | +0.178 | -598 | +502 | 1.59× | 1.31× |

Despite visiting more relevant windows, v2 is worse. Router recall rises from
85.13% in v1 to 88.79% at B=64 and 98.08% at B=128, but the extra coverage
does not translate into better interval sets. This isolates the likely
bottleneck: local calls receive too little temporal evidence, and repeated
autoregressive calls add substantial overhead.

## Comparison with controlled 2 FPS

| Candidate | Δ C-Acc | Δ EtF1 | Δ tIoU | Δ card. error | Frame reduction | GPU reduction | Wall reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual B=64 | -1.25 | -6.46 | -12.51 | +0.009 | 86.0% | 77.4% | 67.2% |
| Residual B=128 | +2.50 | -1.72 | -5.26 | -0.053 | 71.5% | 61.2% | 45.9% |

B=128 remains an efficiency tradeoff relative to 2 FPS: it improves
cardinality accuracy slightly and is much cheaper, but localization and EtF1
remain worse. It is not an overall accuracy improvement.

## What the ablations show

1. **Residual selection is not helping.** At B=64, score and residual ordering
   have identical C-Acc and EtF1, while score ordering is 0.22 tIoU better. At
   B=128, residual ordering changes C-Acc by only +0.31 and EtF1 by +0.12.
   These tiny aggregate differences require paired predictions before they can
   be interpreted.
2. **The stopping mechanism is inert.** No sample exits early at B=64. Only
   four residual samples exit early at B=128, saving seven calls, 76 frames,
   1.97 GPU-seconds, and 2.26 wall-seconds across the entire benchmark.
3. **Most windows are already visited.** The residual policy visits 85.4% of
   windows at B=64 and 95.6% at B=128. Consequently, ordering has little room
   to matter and the procedure approaches exhaustive window processing.
4. **Coverage is not sufficient.** At B=128 the router covers 98.08% of
   occurrences, close to the 99.21% same-call-count oracle, yet EtF1 is only
   16.07. Router failure is no longer the primary limitation.
5. **Call fragmentation is expensive.** B=128 uses 1,203 localizer calls,
   compared with 701 for v1, while processing slightly fewer total frames.
   GPU time nevertheless increases by 58.7%.

## Reviewer-style conclusion

The experiment weakens the proposed algorithmic claim. The deterministic
residual heuristic neither improves the set prediction nor produces a useful
anytime policy. Its stopping condition almost never activates, and its higher
router recall exposes a downstream resolution/fusion bottleneck.

The publishable result remains the v1 controlled inference study. V2 is useful
as evidence that maximizing candidate-window coverage per se is insufficient:
temporal grounding needs a better allocation between broad verification and
dense boundary localization.

Before another full run, replace equal-share local calls with a two-stage
policy:

- use a cheap verification call to reject windows;
- reserve most frames for dense refinement of verified windows;
- cap localizer calls explicitly;
- stop using remaining-window evidence rather than prediction stability alone.

The raw v2 `predictions.jsonl`, `residual_routes.jsonl`, `summary.json`, and
`config.json` should be copied into `results/omtg_residual_search/` before
running paired bootstrap analysis.
