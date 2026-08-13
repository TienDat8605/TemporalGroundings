# Proposal: Boundary-Guided Temporal Sparsification

## Thesis

**Boundary-Guided Sparsification (BGS)** is a training-free augmentation for pretrained video temporal grounders. It allocates temporal evidence toward query-conditioned appearance and disappearance changes, pairs those changes into candidate intervals, and leaves final timestamp prediction to an unchanged frozen model.

## Motivation

Uniform temporal sampling gives the same detail to long irrelevant regions, interval interiors, and the regions most likely to determine IoU: where the queried visual state appears and disappears. Query-similarity peak selection is also insufficient because it can retain an event interior while losing the contrast that distinguishes its start and end.

The method is intended for interval queries such as `dog paw` and `man holding a bottle`, where the target may be a persistent visual-presence state rather than a brief motion event.

## Method

```text
8-second frozen semantic scout
  -> absent / present / uncertain temporal states
  -> directional transition brackets
  -> chronological start/end pairing
  -> local binary bracket refinement
  -> one role-aware frozen grounding call
```

### Coarse presence states

Use one frozen visual observation per eight-second cell. Compute query scores, aggregate Qwen's spatial rows at each timestamp with a top-quartile mean, then normalize within the query-video timeline using median/MAD.

```text
absent:    z <= 0
present:   z >= 1
uncertain: otherwise
```

Require two adjacent observations to confirm a state. This limits false transitions caused by one salient patch, blur, occlusion, or a transient score fluctuation.

### Pairing and refinement

An `absent -> present` change creates a start bracket. A `present -> absent` change creates an end bracket. Pair them in chronological order and score their interval from directional boundary confidence, sparse inside evidence, and outside contrast. Retain up to six non-overlapping pairs for multi-occurrence queries.

Refine only selected brackets. The maximum number of binary rounds is derived from coarse width and target one-second resolution:

\[
K=\left\lceil\log_2(\Delta/1\text{ second})\right\rceil.
\]

An eight-second bracket therefore uses at most three rounds. An uncertain midpoint receives one pair of quarter-point probes. Persistent ambiguity remains a temporal corridor, not a fabricated exact boundary.

### Final frozen grounding

The final evidence pack contains global timeline anchors, paired start/end evidence, sparse interior witnesses, and unresolved-boundary context. It reuses the existing HMVE packer and 12.5% post-encoder retention. The frozen Qwen or UniVTG backend predicts final spans once; router scores never rerank the prediction.

## Compute Contract

For a video duration \(D\), all scout, refinement, and supplementary frames share:

\[
B(D)=\max(64,\operatorname{even}(\lceil D/8\rceil+64)).
\]

The budget grows for long videos because fixed eight-second coverage cannot coexist with a fixed 64-frame budget on a 17-minute video. The only valid baseline is a uniform policy using the identical duration-scaled budget, model, and post-encoder retention.

## Evaluation

Use OMTG as the primary test because it exposes repeated occurrences and makes pairing testable. Compare BGS with a matched `uniform-budget` baseline. Report existing multi-span metrics, candidate coverage diagnostics, paired/unresolved boundary rates, requested-frame ledger, retained evidence, call count, cold wall time, and duration-stratified results.

## Claim and Falsification

**Claim:** at a matched duration-scaled evidence budget, query-conditioned transition pairing improves interval localization or the quality-IoU trade-off over uniform temporal evidence for frozen grounders.

**Falsification:** if paired boundary evidence fails to improve coverage or IoU over matched uniform evidence, BGS remains an interpretable diagnostic policy rather than a temporal-grounding contribution.

## Limits

- A one-view eight-second scout can miss events between samples.
- Frozen similarity is not a calibrated presence probability.
- One-FPS labels do not support sub-second accuracy claims.
- The method cannot claim an end-to-end speedup unless cold scout and refinement costs are included.

## Intended Repository Location

When documentation editing is permitted, move this file to:

`hybrid_vtg/docs/research-notes/idea2/proposal_boundary_guided_sparsification.md`
