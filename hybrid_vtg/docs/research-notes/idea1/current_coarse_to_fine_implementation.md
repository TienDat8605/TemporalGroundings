# Current coarse-to-fine implementation

**Status:** implemented baseline, as of 2026-08-12  
**Method name:** `coarse-to-fine-64`  
**Source:** [`src/hybrid_vtg/methods/coarse_to_fine_64/__init__.py`](../../../src/hybrid_vtg/methods/coarse_to_fine_64/__init__.py)

## Purpose and scope

The current method is a training-free, scene-window coarse-to-fine search with a strict budget of 64 decoded source frames per query. It first spends part of the budget representing and ranking coarse temporal windows, then spends the remainder grounding the query independently inside the highest-ranked windows.

This note describes the code that exists now. It is distinct from the proposed semantic-corridor method: the current implementation can select disconnected windows, makes one grounding call per selected window, and has no uncertainty-gated full-video fallback.

## End-to-end pipeline

```text
video + text query
  -> query-independent scene windows
  -> enforce a feasible 64-frame routing/grounding budget
  -> frozen Qwen3-VL-Embedding-2B ranks all routed windows
  -> select the top-K windows
  -> divide the remaining frames across those windows
  -> frozen grounding backend predicts locally in each window
  -> convert window-relative timestamps to absolute video time
  -> fuse cross-window duplicates for multi-occurrence queries
  -> return predictions in chronological order
```

The method is model-agnostic at the interface level: it calls a `ModelBackend` to encode timestamps and predict spans. Our main experiment uses the frozen Qwen3-VL-4B backend, optionally with Mage encoder pruning and/or SemVID post-encoder pruning. These pruning choices operate inside each local grounding call and do not alter temporal routing or frame allocation.

The two pruning policies and their sequential interaction are documented in [Current Mage and SemVID pruning implementation](current_pruning_implementation.md).

## 1. Construct query-independent temporal windows

Window construction runs before the grounding model is loaded and is shared by all queries for the same video revision.

The primary policy uses PySceneDetect's `ContentDetector` with:

- threshold `27.0`;
- `frame_skip=4`;
- the headless OpenCV backend.

Detected scene boundaries are converted into windows by greedily choosing the farthest boundary between 20 and 60 seconds from the current cursor. If no valid boundary exists, the next endpoint is 45 seconds later. A final remainder shorter than 20 seconds is absorbed into the preceding window. From the second window onward, the start is moved two seconds earlier to provide overlap; a window is clipped back to at most 60 seconds if necessary.

If scene detection fails or returns no usable boundary, the deterministic fallback uses 45-second windows with four seconds of overlap, hence a nominal 41-second hop. It also absorbs a final tail shorter than 20 seconds.

For a video that produces only one window, the method bypasses routing and gives all 64 uniformly sampled frames to one full-window grounding call.

### Scene cache

Scene windows are cached under the method cache's `scenes/` directory. The key includes the resolved video path, file size, modification time, recorded duration, cache schema, and scene policy. It does not include the query, model, pruning policy, seed, or run. Consequently, repeated annotations and matched model variants reuse the same scene analysis.

## 2. Make the 64-frame budget feasible

Let $N$ be the number of routed windows after any coalescing. For $N>1$, the number of windows retained for local grounding is

\[
K = \min\left(N, 8, \max\left(2, \left\lceil\sqrt{N}\right\rceil\right)\right).
\]

Thus, at least two windows are selected when routing is active, the selection grows sublinearly with the number of candidates, and no more than eight local grounding calls are made.

Each routed window receives the same number $r$ of router frames, where the implementation chooses the largest integer in $\{1,2,3,4\}$ satisfying

\[
rN + 2K \le 64
\]

and requiring $rN$ to be even. The second constraint leaves an even grounding budget, while $2K$ reserves at least two local frames per selected window.

If the original number of windows makes this impossible, adjacent chronological windows are coalesced into a smaller number of contiguous groups until a feasible $N$ is found. Coalescing uses approximately equal groups of window indices; each new window spans from the first start to the last end in its group.

The exact budget split is

\[
B_{\text{router}} = rN, \qquad
B_{\text{local}} = 64-rN.
\]

The local budget is divided as evenly as possible among the $K$ selected windows in units of two frames. Any extra two-frame units go to the earlier windows in router-ranked order. The implementation asserts at runtime that

\[
B_{\text{router}} + \sum_{i=1}^{K} B_i = 64.
\]

Here, “frame budget” means sampled/decoded source frames used by the router and grounder. It is not a guarantee of equal pixel count, visual-token count, model calls, latency, or energy relative to another method.

## 3. Rank windows with frozen video-text embeddings

The router is frozen `Qwen/Qwen3-VL-Embedding-2B`, loaded through Sentence Transformers and deliberately placed on CPU so that it does not compete with the 4B grounder for GPU memory.

For every routed window, the method uniformly samples exactly $r$ timestamps, decodes those frames to JPEG, and passes the complete frame list as one video input. Processor-side video sampling is disabled, so the embedding model cannot silently change the temporal budget. The query and video inputs use the retrieval prompts defined in the implementation.

Both text and video embeddings are L2-normalized. A window's router score is therefore their dot product, equivalent to cosine similarity:

\[
s_j = \hat{v}_j^\top \hat{q}.
\]

The top $K$ windows are selected by descending score, with the chronological window index as the deterministic tie-breaker.

### Embedding cache

Query embeddings and video-window embeddings are cached separately:

- the query key includes the exact query text, model ID, prompts, schema, and policy version;
- the video key includes the video revision, duration, routed boundaries, exact sampled timestamps, frames per window, model ID, prompts, schema, and policy version.

This separation lets a new query reuse expensive query-independent video embeddings, while matched dense/Mage/SemVID runs can normally reuse both sides. Cache entries are shape- and finite-value-checked; invalid entries are recomputed and replaced atomically.

## 4. Ground each selected window independently

For selected window $W_i=[a_i,b_i]$ with allocation $B_i$, the method samples $B_i$ uniform timestamps in that window and calls:

```text
evidence_i = backend.encode(video, timestamps_i)
prediction_i = backend.predict(query, evidence_i, context=[a_i, b_i])
```

The Qwen backend prompts in window-relative seconds, from zero to $b_i-a_i$. It then adds $a_i$ to parsed boundaries, producing absolute video timestamps. Invalid out-of-video regions are clipped by post-processing.

Each selected window is processed separately. There is no joint attention or evidence exchange across windows, and router frames are not reused as grounder frames. If Mage or SemVID is enabled, its retention ratio is applied within each local call while preserving the same source-frame allocation.

## 5. Aggregate local predictions

Aggregation depends on the dataset's cardinality contract.

### Single-occurrence queries

Selected windows are visited in descending router-score order. The final output is the first predicted span from the first selected window that returns a non-empty prediction. Later windows do not compete with or refine that span.

### Multi-occurrence queries

Every valid local span becomes a candidate carrying its source-window index and router score. Cross-window duplicate fusion then works as follows:

1. Seed candidates in descending router-score order, with deterministic tie-breakers.
2. Compare the seed with candidates from every other source window.
3. From each other window, admit at most one candidate: the one with the highest temporal IoU with the seed, provided that IoU is strictly greater than `0.6`.
4. Never fuse two predictions from the same source window. This preserves distinct nearby occurrences emitted by one local grounding call.
5. Leave adjacent, disjoint, and exactly-`0.6`-IoU candidates separate.

For a fusion group $G$, router scores determine softmax weights

\[
w_i = \frac{\exp(s_i)}{\sum_{j\in G}\exp(s_j)}.
\]

The output boundaries and local confidence are weighted averages:

\[
\hat{t}_s=\sum_{i\in G}w_i t_{s,i},\qquad
\hat{t}_e=\sum_{i\in G}w_i t_{e,i},\qquad
\hat{c}=\sum_{i\in G}w_i c_i.
\]

Finally, fused and unfused spans are sorted chronologically. Fusion is provenance-aware duplicate suppression, not temporal NMS over all predictions.

## Telemetry and invariants

Every prediction records:

- window source (`content` or `uniform-fallback`) and bypass state;
- routed windows, router scores, selected indices, and frame allocations;
- query/video embedding cache hits and cache paths;
- router, per-window encoder, and per-window prediction times;
- dense and retained encoder evidence units;
- SemVID input and retained evidence units when enabled;
- LLM input/output token counts;
- local and absolute spans, raw local outputs, and fusion provenance;
- router, grounder, and total source-frame counts.

Tests enforce the exact 64-frame total, even local allocations of at least two frames, cache reuse and invalidation, preservation of pruning telemetry, relative-to-absolute timestamp behavior through the backend contract, and the cross-window fusion rules.

## What the current method does not implement

The following ideas appear elsewhere in the research notes but are not part of this implementation:

- no route-confidence or stability gate;
- no fail-open full-video fallback for uncertain long-video routes;
- no single continuous semantic corridor around selected evidence;
- no adaptive halo around selected windows beyond the fixed overlap created during window construction;
- no joint query-level grounding call across selected windows;
- no reuse of router frames by the local grounder;
- no high-FPS endpoint refinement stage;
- no learned component, calibration, or task-specific grounding head.

These distinctions matter when interpreting results. The current implementation establishes a strict fixed-frame embedding-window baseline; it should not be reported as the proposed budget-conserving semantic-corridor method.

## Compact pseudocode

```python
windows = cached_scene_windows(video)

if len(windows) == 1:
    return ground(uniform_sample(windows[0], 64))

windows, policy = make_budget_feasible(windows, total=64)
scores = embed_and_rank(query, windows, policy.router_frames_per_window)
selected = top_k(scores, policy.selected_windows)
allocations = distribute_even_pairs(policy.local_budget, len(selected))

local_predictions = []
for window, frames in zip(selected, allocations):
    evidence = backend.encode(uniform_sample(window, frames))
    local_predictions.append(backend.predict(query, evidence, context=window))

if cardinality == "single":
    return first_nonempty_first_span(local_predictions)
return chronological(fuse_cross_window_duplicates(local_predictions, threshold=0.6))
```

## Related notes

- [Current Mage and SemVID pruning implementation](current_pruning_implementation.md)
- [Current progress report](current_progress_report.md)
- [Temporal sparsification failure diagnosis](temporal_sparsification_failure_note.md)
- [Budget-Conserving Semantic Corridor proposal](proposal_budget_conserving_semantic_corridor.md)
- [Original implementation plan](implementation_plan.md)
