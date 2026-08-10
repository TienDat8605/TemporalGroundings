# Next implementation: safe temporal routing and abstention-aware grounding

## Status

This is the next implementation after the first TACoS diagnostic. It supersedes the old low-confidence policy of spreading eight short expert clips uniformly over the video. It does not supersede the deferred boundary-evidence corridor in `future_boundary_evidence_corridor.md`.

The method remains absolutely training-free: every encoder and VideoLLM is frozen, and no benchmark labels are used to fit a threshold, scorer, or prompt.

## Evidence motivating the change

On the same first ten TACoS samples:

| Configuration | R@1 IoU 0.3 | R@1 IoU 0.5 | mIoU | Seconds/sample |
|---|---:|---:|---:|---:|
| Current temporal router + SemVID | 0.00 | 0.00 | 0.027 | 30.71 |
| Full-video SemVID upper bound | 0.90 | 0.60 | 0.595 | 11.41 |

The full-video run also obtained 100% router containment and a 2.58-second boundary MAE. SemVID retained 12.5% of visual tokens, so spatial pruning itself preserved enough evidence for good TACoS grounding.

The routed run failed for three concrete reasons:

1. Flat SigLIP scores activated a uniform fallback containing eight disconnected components. This retained 43.3% of the duration but lost at least one endpoint in 70% of samples.
2. Qwen was forced to return an interval for every component, including components where the queried event was absent.
3. Final selection used coarse SigLIP boundary contrast and tightness. It discarded locally correct Qwen predictions, such as `[27, 32]` for the target `[26.53, 33.54]`, in favor of unrelated intervals near the end of the video.

Eight separate expert calls were also slower than one continuous full-video SemVID call. Temporal pruning is therefore conditional computation, not a mandatory operation: it must be bypassed when retrieval evidence is unreliable.

## Implementation

### 1. Fail open when temporal retrieval is uncertain

If the frozen temporal retriever's score concentration is below the declared confidence threshold, emit one continuous component covering the complete video:

```text
uncertain retrieval -> [0, video duration] -> one SemVID/Qwen call
```

Do not distribute short windows uniformly. Uniform disjoint coverage is unsuitable for short events because it cannot guarantee target or endpoint containment, and it multiplies expert-model generation overhead.

The manifest and output retain `low_confidence_fallback=true`, making fallback frequency and cost observable. Full-video fallback is expected to retain accuracy rather than claim temporal compute savings.

### 2. Allow the frozen grounder to abstain

For a confidently routed component, Qwen must first decide whether the described event is visibly present. Its deterministic output protocol becomes:

```json
{"present": false, "confidence": 0.07}
```

or:

```json
{"present": true, "confidence": 0.88, "start": 27.0, "end": 32.0}
```

Positive timestamps remain in original-video seconds. Explicit negative components are removed from proposal selection and recorded as component rejections. Legacy `{start, end}` responses remain parseable for compatibility.

The confidence is a frozen-model inference signal, not a learned calibration. It must be reported and ablated; it must not be described as calibrated probability.

### 3. Select proposals lexicographically

Among positive component predictions, rank by:

1. Qwen event-presence confidence;
2. query-evidence boundary contrast and interval tightness;
3. coarse component retrieval score;
4. deterministic earliest-time tie break.

This prevents weak coarse boundary fluctuations from overriding the expert model's event-presence decision. Boundary quality remains useful only after semantic verification.

### 4. Preserve absolute timestamp semantics

Qwen already follows the original-video timestamp prompt on TACoS. `absolute` remains the default and required benchmark setting. Clip-relative conversion is an explicit compatibility option, not an automatic TACoS fix.

## Immediate ablations after implementation

Use fresh output manifests and the same fixed sample IDs:

1. full-video SemVID upper bound;
2. safe fallback + SemVID;
3. confident routing with abstention disabled;
4. confident routing with abstention enabled;
5. presence-first versus boundary-only proposal selection.

Report fallback rate, number of expert components per query, component rejection rate, oracle component-prediction recall, final grounding accuracy, decoded frames/pixels, retained tokens, and wall time.

## Success gates

Before implementing codec-assisted pruning or boundary-evidence corridors:

- uncertain queries must match the full-video SemVID path exactly apart from coarse-search overhead;
- final selection must no longer discard correct positive component predictions for a lower-presence proposal;
- routed target full-containment should be measured separately on non-fallback samples;
- the hybrid method must outperform full-video SemVID in compute on the subset where it actually prunes time;
- aggregate accuracy must remain competitive with the full-video upper bound.

If confident frozen retrieval cannot meet these gates, temporal pruning should remain an optional fast path rather than the claimed default for that benchmark.
