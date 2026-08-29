# Idea 3 Run Results

## Scope and Interpretation

These results were read from the run artifacts under [results/runs](../../../results/runs). All completed runs use seed 42 and batch size 1. The QVHighlights runs use `MCG-NJU/TimeLens2-4B`; the OMTG runs use either TimeLens-8B or TimeLens2-4B, with `sgde-64` unless marked Native or otherwise noted. Percentages below are converted from the stored fractional metrics. A run's metrics scope counts all requested samples; failed samples would count as empty predictions.

## QVHighlights-TimeLens

| Run | Samples | mIoU | R@1 0.3 | R@1 0.5 | R@1 0.7 | Status |
|---|---:|---:|---:|---:|---:|---|
| Native TimeLens2-4B, p010 | 155 | 57.663% | 74.194% | 63.226% | 48.387% | complete |
| SGDE-64 + SigLIP2-Base, p010 | 155 | **58.419%** | **76.774%** | 63.226% | 47.742% | complete |
| Native TimeLens2-4B, p100 | 1,541 | **64.689%** | **78.196%** | **70.214%** | **54.899%** | complete |
| SGDE-64 + SigLIP2-Base, p100 | 1,541 | 57.908% | 72.615% | 61.778% | 47.956% | complete |

Additional QVHighlights artifact status:

- Native p001: 16 samples, 44.655% mIoU, 50.000% / 50.000% / 37.500% R@1.
- SGDE-64 + SigLIP2-Base p001: 16 samples, 30.865% mIoU, 43.750% / 25.000% / 12.500% R@1.
- `qvhighlights-timelens--timelens2-4b--sgde-64--seed-42` has a manifest but no metrics or predictions file, so it is reported as manifest-only and excluded from comparisons.

## OMTG Scout Comparison: TimeLens2-4B

All four OMTG runs are complete on 320 samples.

| Scout variant from run name | tIoU | tF1 0.3 | tF1 0.5 | tF1 0.7 | tP 0.3 | tR 0.3 | C-Acc | EtF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nemotron-1B | **44.059%** | **52.342%** | **41.574%** | **25.947%** | **73.137%** | **44.697%** | **22.813%** | **14.983%** |
| SigLIP2-Base | 29.868% | 38.990% | 31.315% | 19.744% | 55.987% | 32.947% | 17.188% | 10.443% |
| SigLIP2-Large | 29.674% | 40.906% | 32.377% | 19.969% | 59.330% | 34.182% | 16.875% | 10.174% |
| SigLIP2-NaFlex | 28.765% | 39.474% | 29.419% | 18.311% | 56.272% | 33.505% | 18.750% | 10.539% |

The OMTG artifacts use the same SGDE configuration: 1 FPS scout, 64-frame grounding budget, six nominal anchors, four seconds of context, one encoder call, and one primary grounder call. The run names identify the scout variant; the shared manifests do not expose a separate `scout_model` field.

## OMTG TimeLens-8B Runs

All complete 8B runs cover 320 samples. The two SGDE-64 rows have identical prediction-file hashes and metrics, although they were recorded under separate standard and multiwindow run directories.

| Run | Proposed? | Budget | C-Acc | EtF1 | tIoU | tF1 0.3 | tF1 0.5 | tF1 0.7 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Native TimeLens-8B | Control | native | 19.688% | 14.226% | **46.565%** | **56.055%** | **47.977%** | **32.695%** | complete |
| SGDE-64 | Idea 3 | 64f | **26.563%** | **16.317%** | 44.265% | 53.759% | 41.591% | 24.788% | complete |
| SGDE-64 baseline | Baseline | 64f | 22.500% | 11.609% | 39.542% | 49.328% | 37.726% | 22.288% | complete |

Additional 8B artifact status:

- SGDE-128 has a manifest and predictions file but no metrics file, so it is manifest-only and excluded from comparisons.
- The standard and multiwindow SGDE-64 runs each have a failed p001 probe (4 requested, 0 successful, 4 failed); their p100 runs are complete.
- The Native p100 run is complete at 320/320 despite two entries in `errors.jsonl`; those errors do not affect its stored completion status or metrics.

## OMTG Baseline and Non-Proposed Methods

The study defines **`uniform-one-shot`** as the fixed-budget baseline: the grounder receives 64 uniformly sampled frames, with no learned or embedding-based routing. The conventional paper-style comparison is **controlled 2 FPS**, which uses duration-derived frame counts and is therefore not a matched 64-frame baseline. The SGDE rows above use the same 320-sample OMTG benchmark and are directly comparable to the fixed-budget rows on metric definitions, but their frame-routing policy is different.

### TimeLens2-4B: complete 320-sample controls

| Method / schedule | Proposed? | Budget | C-Acc | EtF1 | tIoU | tF1 0.3 | tF1 0.5 | tF1 0.7 | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform one-shot | Baseline | 64f | 11.875% | 8.372% | 41.760% | 38.919% | 27.090% | 16.150% | 427.1s |
| SGDE-64 + Nemotron-1B | Idea 3 | 64f | 22.813% | 14.983% | **44.059%** | **52.342%** | **41.574%** | **25.947%** | current run |
| SGDE-64 + SigLIP2-Base | Idea 3 | 64f | 18.750% | 10.443% | 29.868% | 38.990% | 31.315% | 19.744% | current run |
| SGDE-64 + SigLIP2-Large | Idea 3 | 64f | 16.875% | 10.174% | 29.674% | 40.906% | 32.377% | 19.969% | current run |
| SGDE-64 + SigLIP2-NaFlex | Idea 3 | 64f | 18.750% | 10.539% | 28.765% | 39.474% | 29.419% | 18.311% | current run |
| Full-video multipass | Control | 64f | 8.750% | 4.601% | 37.198% | 28.175% | 15.690% | 7.623% | 532.8s |
| Uniform-window-local | Control | 64f | 34.063% | 16.399% | 35.231% | 42.730% | 33.217% | 19.716% | 604.4s |

### Conventional 2 FPS and larger-budget controls

| Model / schedule | Proposed? | Samples | C-Acc | EtF1 | tIoU | tF1 0.3 | tF1 0.5 | tF1 0.7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TimeLens2-4B, controlled 2 FPS | Conventional baseline | 320 | 24.375% | 17.793% | **50.295%** | 53.250% | 43.243% | 29.102% |
| TimeLens2-4B, uniform one-shot 128f | Control | 320 | 20.000% | 14.464% | 46.744% | 47.207% | 36.947% | 23.191% |
| TimeLens2-4B, embedding-window-local 128f | Routing comparator | 320 | 33.125% | **21.144%** | 46.583% | **54.933%** | 42.720% | 28.485% |
| Qwen3-VL-4B-Instruct, uniform one-shot 64f | Non-proposed model control | 320 | 1.250% | 1.042% | 28.055% | 25.868% | 19.352% | 13.007% |
| Qwen3-VL-4B-Instruct, embedding-window-local 64f | Routing comparator | 320 | 22.500% | 13.611% | 37.220% | 41.179% | 31.509% | 20.008% |

The 2 FPS rows use substantially more frames than the 64-frame experiments and should be read as quality references, not matched-budget baselines. The legacy comparison reports that TimeLens2 embedding-window-local at 64f improves C-Acc and EtF1 over uniform one-shot, while full-video multipass is worse than the one-shot control.

## Takeaways and Caveats

- On OMTG, Native TimeLens-8B has the strongest recorded tIoU and tF1 values, while SGDE-64 with TimeLens-8B has the strongest C-Acc and EtF1 among the fixed 64-frame rows.
- SGDE's strongest documented QVHighlights result is the 155-sample slice, where it slightly beats Native on mIoU and low-IoU recall.
- That advantage does not transfer to the full 1,541-sample QVHighlights run.
- Among the TimeLens2-4B OMTG scout variants, Nemotron-1B is substantially stronger than all three SigLIP2 variants under the same recorded SGDE budget.
- The 8B SGDE-64 result improves C-Acc and EtF1 over the 8B baseline, but its tIoU and tF1 remain below Native 8B.
- The OMTG runs are not directly comparable to QVHighlights because they use different metric definitions.