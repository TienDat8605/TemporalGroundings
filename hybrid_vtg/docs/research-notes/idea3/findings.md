# Findings:

## Scope and framing

This note summarizes the current evidence for the OMTG benchmark and distinguishes clearly between the proposed method and the baselines.

The true native OMTG baseline is the TimeLens2-4B run under the benchmark’s standard controlled 2 FPS setting, without SGDE routing or scouting. For completeness, we also include the current native TimeLens-8B, but the primary benchmark comparison remains the TimeLens2 on the OMTG set.

## Contributions

1. Baseline correction for OMTG
   - We corrected the OMTG baseline prompt and evaluation setup so that the comparison reflects the benchmark task more faithfully and is not distorted by an instruction mismatch.
   - In the legacy experiments, the benchmark effectively asks for multi-span grounding while the official prompt was singular in form; correcting the baseline is therefore essential for a fair comparison with the proposed method.

2. Training-free Scouting method
   - We proposed a training-free video grounding pipeline that uses a cheap global scout to estimate relevance over the full video, then concentrates the expensive dense evidence on the most promising interval.
   - In practical terms, the method reduces the dense evidence burden from a full-video `N`-fps pass to a scout + concentrated verification path, which brings the effective token cost down to roughly half of the dense route for the matched 64-frame setting.
   - This enables a better compute-quality tradeoff: the method keeps a fixed 64-frame (or 128) grounding budget while improving event localization and cardinality consistency relative to the naive fixed-budget baseline.
   - The results are still competitive with stronger trained-on-dataset models in terms of the overall accuracy-efficiency tradeoff, even though the native model remains the upper bound on strict temporal IoU metrics.

3. Preprocessing for hot-starting and faster inference
   - The scout stage produces reusable relevance signals and cached features that can be reused across runs or successive passes.
   - This provides a practical hot-start benefit: inference time is reduced because we avoid redoing the full scouting pipeline from scratch for repeated or closely related queries.
   - In other words, the method is not only about better final grounding but also about making the full pipeline more operationally efficient in repeated settings for the same video.

## Experiments included

### OMTG experiments

The comparison set below uses the full OMTG benchmark at 320 samples unless noted otherwise.

Included runs:

- Legacy OMTG native baseline: TimeLens2-4B, controlled 2 FPS
- Legacy OMTG fixed-budget controls: uniform one-shot 64f;
- Current SGDE proposals: TimeLens-8B SGDE-64; TimeLens2-4B SGDE-64 + Nemotron-1B; TimeLens2-4B SGDE-64 + SigLIP2-Base; TimeLens2-4B SGDE-64 + SigLIP2-Large; TimeLens2-4B SGDE-64 + SigLIP2-NaFlex
- Current native dense baseline: TimeLens-8B native

We exclude the legacy HMVE and coarse-to-fine-64 results as they are not part of the retained comparison set and are less directly aligned to the current SGDE evaluation.

## OMTG master comparison table

The table below combines the main OMTG native and non-SGDE controls with the SGDE proposal variants. This is the comparison set we view as the primary evidence for method quality and efficiency.

| Method | Model / setup | Budget | C-Acc | EtF1 | tIoU | tF1@0.3 | tF1@0.5 | tF1@0.7 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Native OMTG baseline | TimeLens2-4B, controlled 2 FPS | duration-derived | 24.38% | 17.79% | 50.30% | 53.25% | 43.24% | 29.10% | true native OMTG baseline |
| Baseline 64 frames | TimeLens2-4B, uniform one-shot | 64f |  |  |  |  |  |  |  baseline |
| SGDE-64 + Nemotron-1B | TimeLens2-4B | 64f | 22.81% | 14.98% | 44.06% | 52.34% | 41.57% | 25.95% | our proposal |
| SGDE-64 + SigLIP2-Base | TimeLens2-4B | 64f | 17.19% | 10.44% | 29.87% | 38.99% | 31.32% | 19.74% | our proposal |
| SGDE-64 + SigLIP2-Large | TimeLens2-4B | 64f | 16.88% | 10.17% | 29.67% | 40.91% | 32.38% | 19.97% | our proposal |
| SGDE-64 + SigLIP2-NaFlex | TimeLens2-4B | 64f | 18.75% | 10.54% | 28.77% | 39.47% | 29.42% | 18.31% | our proposal |
| Native dense baseline | TimeLens-8B, native | duration-derived | 19.69% | 14.23% | 46.57% | 56.05% | 47.98% | 32.69% | older dense baseline |
| SGDE-64 | TimeLens-8B | 64f | 26.56% | 16.32% | 44.27% | 53.76% | 41.59% | 24.79% | our proposal |

### Important observations from the master table

- The true native OMTG baseline is the legacy TimeLens2-4B controlled 2 FPS run, with tIoU 50.30% and tF1@0.5 43.24%.
- The strongest SGDE result is the TimeLens-8B SGDE-64 run, with C-Acc 26.56% and EtF1 16.32%.
- The strongest TimeLens2-4B SGDE variant is the Nemotron-1B scout, with C-Acc 22.81% and EtF1 14.98%.
- Across the direct 64-frame comparison, SGDE improves over the naive uniform one-shot baseline and remains competitive with the stronger native dense baseline on the summary metrics that matter for event-level grounding.

## Conclusions
### 1. SGDE improves the fixed-budget regime over the naive baseline

Relative to the TimeLens2-4B uniform one-shot 64f baseline, the SGDE-64 variant improves the key summary metrics in the fixed-frame regime:

- C-Acc: 22.81% vs []%
- EtF1: 14.98% vs []%
- tIoU: 44.06% vs []%
- tF1@0.5: 41.57% vs []%

The same pattern holds for the TimeLens-8B SGDE-64 run against the 64-frame baseline: SGDE improves C-Acc and EtF1 while preserving strong localization quality under a much tighter compute budget.

### 2. The native dense model still has the edge on strict overlap metrics

The native TimeLens2-4B controlled 2 FPS baseline remains the strongest on tIoU and thresholded localization metrics. This is expected: dense inference has access to more evidence, and a native dense model is not constrained by the same routing or scout budget.

However, the SGDE design is valuable exactly because it provides a better compute-quality tradeoff. It does not aim to beat the dense model on every metric; it aims to preserve most of the performance while reducing the number of frames and tokens processed.

### 3. Scout quality matters

The Nemotron-1B scout is consistently the best TimeLens2-4B SGDE variant under the same 64-frame setup. The SigLIP2 variants are weaker, which indicates that the scout is not a trivial add-on but a central determinant of the final routing quality.

This is an important result: the proposal’s gain is not only in the downstream verification stage, but also in the quality of the initial global relevance estimate.

### 5. SGDE remains practically useful despite not being the absolute best on every metric

The overall conclusion is that SGDE is a meaningful efficiency method rather than a strict state-of-the-art replacement for the dense baseline. In practical terms, it offers:

- better accuracy than naive fixed-budget baselines,
- better event-level coverage and cardinality consistency,
- a reduced compute footprint compared with dense native evaluation,
- and a hot-start benefit from reusable scouting and preprocessing.

This makes SGDE a credible proposal for resource-constrained long-video grounding, especially when the goal is to preserve grounding quality under a limited token budget rather than to maximize raw localization metrics at any cost.

## QVHighlights note

The QVHighlights results are useful supporting evidence, but the primary decision point remains OMTG. In the current artifacts, the strongest QVHighlights result is the native TimeLens2-4B run on the smaller subset, while the full-sample trend is more mixed. That makes the OMTG fixed-budget comparison the more reliable indicator for the SGDE proposal itself.

## Final assessment

The evidence supports the following summary:

- SGDE improves over the naive fixed-budget non-SGDE baseline and provides a clearer efficiency-quality tradeoff.
- The native dense model remains the strongest absolute performer on overlap metrics, but SGDE is a meaningful and practically useful compromise between accuracy and compute.
- The strongest fixed-budget result in the current OMTG comparison is the TimeLens-8B SGDE-64 run, while the strongest TimeLens2-4B SGDE variant is the Nemotron-1B scout configuration.
