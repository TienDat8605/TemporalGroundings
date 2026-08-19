# OMTG full-split report: Mage 0.8 + SemVID 0.25

**Run status:** complete  
**Benchmark:** OMTG  
**Evaluation size:** 320 queries (100% subset)  
**Seed:** 42  
**Method:** `coarse-to-fine-64`  
**Pruning:** Mage at vision layer 0 with 80% encoder retention, followed by SemVID with 25% final dense-relative retention  
**Models:** TimeLens2-4B, TimeLens-8B, and UniTime

## Executive result

UniTime is the strongest temporal localizer under this fixed routing and pruning configuration. It obtains the best tIoU, temporal precision, recall, and F1 at every evaluated IoU threshold, as well as the best joint EtF1.

TimeLens2-4B is the strongest cardinality estimator. It predicts the exact number of occurrences on 35.31% of queries and has the lowest mean absolute cardinality error, but its predicted temporal intervals are substantially less accurate than UniTime's.

TimeLens-8B does not improve on TimeLens2-4B overall. Its localization scores are nearly tied with TimeLens2-4B, while its cardinality accuracy and EtF1 are worse.

These runs compare model backends under one fixed hybrid pipeline. They do not by themselves measure the benefit of Mage or SemVID, because dense, Mage-only, and SemVID-only controls are not included.

## Run configuration

All three models were evaluated with:

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model <model> \
  --method coarse-to-fine-64 \
  --subset 100 \
  --seed 42 \
  --encoder-pruning mage \
  --encoder-retention 0.8 \
  --encoder-prune-layer 0 \
  --post-pruning semvid \
  --post-retention 0.25 \
  --rerun
```

For each selected local temporal window, Mage targets approximately 80% of the original dense visual cells before the first vision-transformer block. SemVID subsequently targets approximately 25% of the same original dense count before LLM prefill. The ratios are dense-relative rather than multiplicative, so the intended evidence progression is approximately:

```text
100% dense cells -> 80% after Mage -> 25% after SemVID
```

All runs completed successfully:

| Model | Requested | Successful | Failed | Status |
|---|---:|---:|---:|---|
| TimeLens2-4B | 320 | 320 | 0 | Complete |
| TimeLens-8B | 320 | 320 | 0 | Complete |
| UniTime | 320 | 320 | 0 | Complete |

Failed samples would have counted as empty predictions, but there were no failures.

## Primary metrics

All values except Cardinality Error and Count are percentages. Higher is better for every metric except Cardinality Error.

| Model | C-Acc ↑ | Card. Error ↓ | EtF1 ↑ | tIoU ↑ | tF1@0.3 ↑ | tF1@0.5 ↑ | tF1@0.7 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TimeLens2-4B | **35.313** | **1.522** | 3.811 | 15.525 | 19.236 | 8.296 | 2.655 |
| TimeLens-8B | 19.063 | 2.134 | 3.116 | 15.917 | 18.384 | 8.322 | 2.631 |
| UniTime | 28.438 | 1.703 | **9.436** | **23.814** | **30.645** | **18.014** | **8.548** |

### Temporal precision and recall

| Model | tP@0.3 | tR@0.3 | tP@0.5 | tR@0.5 | tP@0.7 | tR@0.7 |
|---|---:|---:|---:|---:|---:|---:|
| TimeLens2-4B | 24.880 | 16.696 | 11.156 | 7.065 | 3.854 | 2.195 |
| TimeLens-8B | 23.500 | 17.302 | 10.653 | 7.903 | 3.469 | 2.467 |
| UniTime | **39.828** | **27.057** | **22.120** | **16.344** | **10.026** | **7.996** |

## Metric interpretation

OMTG is evaluated as a multi-occurrence grounding task.

- **C-Acc** is the percentage of queries for which the number of predicted spans exactly equals the number of target spans.
- **Cardinality Error** is the mean absolute difference between predicted and target occurrence counts.
- **tIoU** computes IoU between the union of predicted intervals and the union of target intervals for each query, then macro-averages over queries.
- **tP/tR/tF1** first use Hungarian matching between predicted and target spans, then count matches whose temporal IoU reaches the stated threshold. The reported values are macro-averages over queries.
- **EtF1** is nonzero for a query only when its predicted cardinality is exactly correct; for such queries it averages tF1 over thresholds 0.3, 0.5, and 0.7. The final value is then macro-averaged. It is therefore not the product of the aggregate C-Acc and aggregate tF1 values.

The exact-cardinality query counts are:

| Model | Exact-count queries | Total | C-Acc |
|---|---:|---:|---:|
| TimeLens2-4B | **113** | 320 | 35.313% |
| TimeLens-8B | 61 | 320 | 19.063% |
| UniTime | 91 | 320 | 28.438% |

## Pairwise findings

### UniTime versus TimeLens2-4B

UniTime trades some cardinality accuracy for much stronger localization:

| Metric | UniTime difference |
|---|---:|
| C-Acc | -6.875 points |
| Cardinality Error | +0.181 occurrences |
| EtF1 | +5.625 points |
| tIoU | +8.289 points |
| tF1@0.3 | +11.410 points |
| tF1@0.5 | +9.718 points |
| tF1@0.7 | +5.894 points |

Although UniTime exactly counts fewer queries than TimeLens2-4B, its EtF1 is 2.48 times as large. This indicates that its correctly counted predictions are also much better aligned temporally.

### UniTime versus TimeLens-8B

UniTime is better on every reported metric:

- C-Acc improves by 9.375 points;
- mean cardinality error falls by 0.431 occurrences;
- EtF1 improves by 6.320 points;
- tIoU improves by 7.897 points;
- tF1 improves by 12.262, 9.692, and 5.917 points at IoU 0.3, 0.5, and 0.7.

### TimeLens2-4B versus TimeLens-8B

The two TimeLens variants have essentially the same localization quality. TimeLens-8B is higher by 0.392 tIoU points and 0.026 tF1@0.5 points, while TimeLens2-4B is higher by 0.852 tF1@0.3 and 0.024 tF1@0.7 points. These differences are too small to treat as meaningful without paired uncertainty estimates.

Their cardinality behavior is not close: TimeLens2-4B improves C-Acc by 16.25 points, lowers mean cardinality error by 0.613 occurrences, and improves EtF1 by 0.695 points. Under this configuration, the larger TimeLens-8B backend offers no clear advantage.

## Diagnostic observations

### 1. UniTime is the best overall model for grounded intervals

UniTime leads both precision and recall at all thresholds, rather than improving one by sacrificing the other. Its tIoU advantage is consistent with its span-matching advantage.

### 2. Accurate occurrence counting remains difficult

Even the best C-Acc is only 35.31%, corresponding to 113 of 320 queries. The best mean cardinality error is 1.522 occurrences. Multi-occurrence enumeration is therefore still a major failure mode for all three models.

### 3. Boundary quality drops sharply at stricter IoU

UniTime's tF1 falls from 30.65 at IoU 0.3 to 8.55 at IoU 0.7. The TimeLens models fall from about 19 to about 2.6. The models can often identify a broadly relevant region but struggle to return sufficiently tight start and end boundaries.

### 4. Precision exceeds recall for all models

At every threshold, temporal precision is higher than temporal recall. This is consistent with conservative or incomplete occurrence recovery, but the aggregate metrics do not reveal whether errors come mainly from underprediction, overprediction on different samples, or router misses. Signed count error and router-recall analysis are needed before assigning a cause.

### 5. EtF1 changes the model ranking

TimeLens2-4B has the best raw cardinality accuracy, yet UniTime has the best EtF1 by a large margin. Exact count alone is insufficient: a model must both enumerate the correct number of events and localize those events accurately.

## Conclusions

Under the full-split OMTG `coarse-to-fine-64 + Mage(0.8, layer 0) + SemVID(0.25)` configuration:

1. **Use UniTime as the strongest localization result:** tIoU 23.814 and EtF1 9.436.
2. **Use TimeLens2-4B as the strongest cardinality result:** C-Acc 35.313% and mean Cardinality Error 1.522.
3. **Do not infer a pruning gain from this comparison:** all models use the same combined pruning, and no unpruned control is present.
4. **Do not infer an efficiency gain from these outputs:** no wall time, GPU time, memory, dense/retained token totals, or cache-state measurements were provided.
5. **Treat boundary precision and occurrence recall as the next diagnostic targets.**

## Required follow-up comparisons

To isolate the effect of each pruning stage, run the same 320-query split and seed for each model with:

| Ablation | Encoder policy | Encoder retention | Post policy | Post retention |
|---|---|---:|---|---:|
| Dense | none | 1.0 | none | 1.0 |
| Mage only | mage | 0.8 | none | 1.0 |
| SemVID only | none | 1.0 | semvid | 0.25 |
| Mage + SemVID | mage | 0.8 | semvid | 0.25 |

The comparison should report accuracy together with decoded frames/pixels, encoder dense and retained units, SemVID input/output units, LLM input tokens, vision time, prefill time, end-to-end time, peak memory, and route/fusion statistics. Paired video-cluster bootstrap intervals are needed before interpreting small model differences.

## Supplied run directories

```text
/root/TemporalGroundings/hybrid_vtg/results/runs/omtg/timelens2-4b--enc-mage-r0.8-l0--post-semvid-r0.25/coarse-to-fine-64/seed-42
/root/TemporalGroundings/hybrid_vtg/results/runs/omtg/timelens-8b--enc-mage-r0.8-l0--post-semvid-r0.25/coarse-to-fine-64/seed-42
/root/TemporalGroundings/hybrid_vtg/results/runs/omtg/unitime--enc-mage-r0.8-l0--post-semvid-r0.25/coarse-to-fine-64/seed-42
```

## Related notes

- [Current coarse-to-fine implementation](current_coarse_to_fine_implementation.md)
- [Current Mage and SemVID pruning implementation](current_pruning_implementation.md)
- [Current progress report](current_progress_report.md)
