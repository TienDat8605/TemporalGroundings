# Current Mage and SemVID pruning implementation

**Status:** implemented, training-free inference policies, as of 2026-08-12  
**Primary source:** [`src/hybrid_vtg/models/pruning.py`](../../../src/hybrid_vtg/models/pruning.py)  
**Qwen integration:** [`src/hybrid_vtg/models/qwen.py`](../../../src/hybrid_vtg/models/qwen.py)

## Summary

The current Qwen evidence backend supports two independent visual-pruning stages:

| Method | Execution point | Selection signal | Main compute affected |
|---|---|---|---|
| Mage-style pruning | Inside the Qwen vision encoder, before a configured vision block | Camera-compensated motion and residual novelty | Later vision-transformer blocks and downstream visual-token processing |
| SemVID pruning | After the vision encoder, immediately before LLM prefill | Query relevance, temporal change, context, and diversity | LLM visual prefill length |

When both are enabled, they execute sequentially:

```text
decoded local-window frames
  -> Qwen patch embedding and positional encoding
  -> Mage selects complete spatial merger cells
  -> remaining Qwen vision blocks encode only Mage survivors
  -> merged visual evidence with timestamps and coordinates
  -> SemVID selects context/object/motion evidence from Mage survivors
  -> compact timestamped visual prompt
  -> frozen Qwen language-model prefill and generation
```

Neither policy changes the coarse-to-fine method's 64 source-frame budget, scene windows, router scores, selected windows, or local frame allocations. Pruning is applied separately inside every selected local grounding call.

Both policies are disabled by default. The matched combined configuration is:

```bash
--encoder-pruning mage \
--encoder-retention 0.5 \
--encoder-prune-layer 0 \
--post-pruning semvid \
--post-retention 0.125
```

## Shared visual representation

The backend first decodes the timestamps allocated to one temporal window. With the normal Qwen configuration, it resizes the frames to a patch-aligned grid whose dense merged-cell count is bounded by `maximum_evidence_units` (default `4096`) for that local call.

Qwen groups neighboring source frames into temporal patch units. Each output visual evidence unit is a spatial merger cell associated with:

- a temporal-unit timestamp;
- a spatial coordinate `(time, row, column)`;
- one Qwen visual embedding after the merger.

Let:

- $T$ be the number of Qwen temporal units;
- $H_c\times W_c$ be the spatial merger-cell grid per temporal unit;
- $D=T H_c W_c$ be the dense visual evidence count before either pruning policy.

This dense count $D$, rather than the number of decoded source frames, is the reference used by both retention ratios.

## 1. Mage-style encoder pruning

### What this implementation is

This is a clean-room, training-free adaptation of Mage-VL's anchor/update principle. It does not use Mage-ViT, a trained codec tokenizer, or copied Mage-VL model code. The implementation currently derives its selection signal from decoded RGB frames; true exported codec motion vectors and residuals could replace that provider later without changing the sparse-encoder interface.

Mage is query-independent. Its purpose is to keep spatial cells that contain motion or newly appearing visual information before most of the expensive vision encoder runs.

### Motion and residual importance

For each consecutive decoded-frame pair, the implementation:

1. converts the frames to grayscale and resizes them to an analysis grid;
2. computes dense Farneback optical flow;
3. estimates global camera motion as the median flow vector;
4. subtracts that global vector and takes the remaining local-flow magnitude;
5. backward-warps the previous frame using the flow and measures absolute luminance residual against the current frame;
6. robustly normalizes positive motion and residual values by their 95th percentiles and clips them to $[0,1]$;
7. assigns equal weight to both signals:

\[
I = 0.5\,I_{\text{local-motion}} + 0.5\,I_{\text{residual}}.
\]

The novelty map is downsampled to the Qwen merger-cell grid. When several decoded frames belong to one Qwen temporal unit, their cell importance is combined by a temporal maximum. The first frame has a zero-change map because it has no predecessor.

These importance maps are cached by the video revision, exact sampled timestamps, and processed grid shape. They are independent of the text query and can be reused across annotations and SemVID variants that use the same local frames.

### Exact cell budget

For encoder retention ratio $r_E$, Mage targets

\[
M = \min\left(D,\max\left(T,\operatorname{round}(D r_E)\right)\right).
\]

The $T$ floor guarantees at least one complete spatial merger cell per temporal unit. Selection is deterministic:

1. **Per-time floor:** keep the highest-importance cell in every temporal unit.
2. **Dense anchors:** beginning at temporal unit zero and then every eight units, retain the complete spatial grid if it still fits the exact global budget.
3. **Sparse updates:** fill all remaining slots with the highest-importance unselected cells across space and time. Ties use temporal and cell index order.

Mage always selects complete Qwen merger cells. For merger size $m$, one retained cell contributes all $m^2$ constituent patch tokens before the merger. It never keeps an incomplete cell.

### Where pruning happens in Qwen

The backend installs a wrapper around Qwen's vision encoder. Qwen first performs patch embedding and constructs its original positional and rotary encodings. Immediately before vision block `encoder_prune_layer`, the wrapper slices:

- hidden states;
- rotary-position tensors;
- per-time cumulative sequence lengths.

It preserves the selected cells' original coordinates instead of compressing them onto a new dense grid. All remaining vision blocks and the Qwen merger then operate on the sparse sequence. The default matched experiment uses layer `0`, so every transformer block sees only the retained cells; patch construction and patch embedding still occur densely.

Choosing a later prune layer would let earlier blocks process the dense sequence and only save work after that layer. The implementation validates the layer index when the model is loaded and currently supports batch size one for this sparse path.

### Mage output

Mage returns a variable number of visual embeddings per temporal unit, together with original timestamps and spatial coordinates. Metadata records the dense count, retained count, per-time counts, retention ratio, prune layer, importance backend, and importance-cache hit.

Mage alone reduces vision-encoder work after the pruning point, but it is not query-aware and does not perform a second token selection before LLM prefill.

## 2. SemVID post-encoder pruning

### What this implementation is

SemVID is adapted from the official Qwen3-VL implementation at the upstream revision recorded in the code and `NOTICE.md`. Our adapter supports different token capacities at different timestamps, which is necessary because Mage can leave a variable number of cells per temporal unit.

SemVID runs on already encoded visual embeddings immediately before Qwen's language-model prompt is assembled. It is query-aware and reduces LLM prefill, but when used alone it does not save vision-encoder computation: the dense visual grid has already passed through Qwen's vision encoder.

### Query and frame-level allocation

The query is embedded token by token using the frozen Qwen language-model input embedding table. Queries of 2–50 tokens retain tokenwise representations; queries longer than 50 tokens are mean-pooled to one vector.

Visual evidence units with the same timestamp form one temporal group. SemVID mean-pools each group to a frame-level/global embedding $g_t$ and computes:

- **query relevance:** the mean of the two largest cosine-like similarities between $g_t$ and the query-token embeddings, or all available similarities if fewer than two exist;
- **motion energy:** the L2 distance between normalized consecutive group embeddings.

The temporal allocation weight is

\[
a_t = 0.7\,\max(0,\operatorname{rel}_t) + 0.3\,\operatorname{motion}_t.
\]

For post-retention ratio $r_S$, available input count $M_{\text{in}}$, and original dense count $D$, the final target is

\[
S = \min\left(M_{\text{in}},\max\left(T,\operatorname{round}(D r_S)\right)\right).
\]

SemVID first reserves one token for every temporal group. It distributes the remaining target across groups in proportion to $a_t$, subject to each group's available capacity. The allocator is deterministic and falls back to uniform weights if all active weights are effectively zero.

### Context, object, and motion roles

Within each temporal group, SemVID fills its assigned token count using three roles:

1. **Context:** always keep the token most similar to that group's mean embedding. This ensures every timestamp remains represented.
2. **Object:** assign approximately 60% of the remaining slots to query-relevant tokens. For tokenwise queries, slots are first distributed across query tokens. Selection uses maximal marginal relevance with $\lambda=0.9$, favoring query relevance while penalizing redundant visual embeddings.
3. **Motion:** assign the remaining slots to changing evidence. When spatial coordinates exist, a token is compared with the same cell in adjacent temporal groups; otherwise it is compared with adjacent group means. For tokenwise queries, the final score is an equal mixture of normalized motion and query relevance. For a pooled query, it uses motion alone.

If any assigned slots remain unfilled, SemVID deterministically uses the largest-norm remaining embeddings as context tokens. Selected evidence is returned in timestamp order, with its context/object/motion roles and surviving spatial coordinates.

The compact evidence is then inserted into the Qwen prompt with relative timestamps. Generation remains deterministic and uses the same localization prompt as the dense backend.

## 3. How Mage and SemVID work together

The two stages are complementary because they prune at different depths:

```text
dense visual cells D
  -- Mage, motion/residual, before vision block --> M survivors
  -- SemVID, query/motion/context, before LLM --> S survivors
```

The order is fixed: Mage always runs during `encode()`, and SemVID runs later during `predict()`. SemVID can select only from Mage's survivors; it cannot restore a cell removed by Mage.

### Retention ratios are relative to the same dense baseline

The combined retention ratios are deliberately not multiplicative. Both refer to the original dense count $D$:

\[
M \approx D r_E, \qquad S \approx D r_S,
\]

with the per-time floors, rounding, and availability caps shown above. The backend requires

\[
r_S \le r_E
\]

when both policies are active, because the desired SemVID output cannot exceed the Mage input budget.

For example, ignoring rounding and temporal floors:

| Stage | Matched ratio | Example with $D=1000$ |
|---|---:|---:|
| Dense Qwen cells | 1.0 | 1000 |
| After Mage | $r_E=0.5$ | 500 |
| After SemVID | $r_S=0.125$ of dense | 125 |

The final count is approximately 12.5% of dense, not $0.5\times0.125=6.25\%$. Equivalently, SemVID keeps about one quarter of the Mage survivors in this matched setting.

The code preserves `dense_evidence_units` through Mage and passes it explicitly to SemVID. SemVID records both its actual input count and its final retained count, making this convention observable in telemetry.

### Information and compute flow

Using both policies changes the signals available at the second stage:

- Mage is query-independent and prioritizes local change/residual evidence plus periodic dense anchors.
- SemVID receives only those survivors, but uses the query to decide which survivors are useful for object identity, motion, and temporal context.
- Original timestamps and cell coordinates survive Mage, allowing SemVID to compare the same spatial location across neighboring temporal units when that location is present in both.
- Because Mage produces variable per-time capacities, SemVID's allocator caps each temporal group's budget by its actual number of survivors.

In compute terms:

- Mage can reduce work in the vision-transformer blocks after its prune layer and reduces the visual evidence passed downstream.
- SemVID further reduces the number of visual embeddings inserted into the LLM prompt and therefore the visual prefill sequence.
- Motion-map computation, decoding, preprocessing, patch embedding, and any vision blocks before the Mage prune layer remain costs of the combined method.
- The two methods do not reduce the number of local grounding calls produced by coarse-to-fine routing.

## Configuration rules

The backend enforces:

- both retention ratios must be in $(0,1]$;
- a non-unit encoder retention requires `--encoder-pruning mage`;
- a non-unit post retention requires `--post-pruning semvid`;
- the Mage prune layer must be non-negative and must exist in the loaded vision encoder;
- when both are enabled, `post_retention <= encoder_retention`.

The four intended ablations are:

```text
Dense:          encoder=none, retention=1.0; post=none, retention=1.0
Mage only:      encoder=mage, retention=0.5; post=none, retention=1.0
SemVID only:    encoder=none, retention=1.0; post=semvid, retention=0.125
Mage + SemVID:  encoder=mage, retention=0.5; post=semvid, retention=0.125
```

These configurations use identical source-frame routing and allocation. They differ only in how much spatial evidence is processed inside the vision encoder and supplied to the LLM.

## Telemetry and tested invariants

Per local window, the current coarse-to-fine telemetry exposes:

- dense encoder evidence units;
- Mage-retained encoder evidence units;
- SemVID input and output evidence units;
- SemVID context/object/motion role counts;
- LLM input and output token counts;
- local encode and prediction latency.

Unit tests verify that:

- Mage respects the exact complete-cell budget and keeps the per-time floor;
- motion/residual importance reacts to localized change;
- pruning at layer zero reduces the sequence before the first vision block;
- SemVID meets its target, retains every timestamp, and preserves chronological order;
- SemVID's combined target is relative to the original dense count rather than the Mage-retained count;
- invalid or inconsistent pruning configurations are rejected;
- coarse-to-fine source-frame totals and temporal routing are unchanged when both policies are active.

## Interpretation limits

- “Mage” in experiment labels means this project's Mage-style clean-room policy, not the original trained Mage-VL system.
- SemVID token retention alone is not an end-to-end speed claim because it does not remove vision encoding, decoding, or repeated local calls.
- Mage's optical-flow/residual analysis adds CPU work and currently operates on decoded frames rather than native codec vectors.
- Both policies preserve at least one cell per temporal unit, so very low requested ratios can be raised by the temporal floor.
- The combined method cannot recover query-relevant evidence removed by the query-independent Mage stage.
- Retention ratios describe visual evidence units, not decoded source frames or pixels.

## Related notes

- [Current coarse-to-fine implementation](current_coarse_to_fine_implementation.md)
- [Current progress report](current_progress_report.md)
- [Codec-assisted SemVID proposal](first_spatial_improvement_codec_assisted_semvid.md)
- [Boundary Evidence Chain SemVID proposal](proposal_boundary_evidence_chain_semvid.md)
