# OMTG inference, prompting, and efficiency

## Research assessment

This report compiles all complete 320-query runs for TimeLens2-4B and Qwen3-VL-4B. Metric deltas use paired 95% percentile-bootstrap confidence intervals clustered by video (10,000 resamples; fixed seed `20260725`). Intervals are exploratory and are not corrected for multiple comparisons.

| Proposed claim | Verdict | Fair interpretation |
| --- | --- | --- |
| Multi-span grounding is also an inference problem | **Supported, with scope limits** | On base Qwen, embedding-window-local raises C-Acc from 1.25 to 22.50 and EtF1 from 1.04 to 13.61 without OMTG-specific training. This shows inference policy matters; it does not show training is unnecessary. |
| Our method improves TimeLens2 over 2 FPS | **Partially supported** | At 64 frames it improves cardinality but loses substantial localization accuracy. At 128 frames it reaches C-Acc 33.12 and EtF1 21.14, while approximately matching the controlled 2 FPS thresholded localization scores. |
| OMTG has a prompt flaw | **Supported as a benchmark confound** | 320/320 official instructions ask for a singular segment, while every label is multi-span. Call this an instruction–annotation mismatch, not evidence that the underlying videos or labels are flawed. |
| Our method is cheaper | **Supported versus 2 FPS only** | Fixed-frame inference uses far fewer frames and less total time than 2 FPS. Against fixed-frame uniform inference, routing can reduce synchronized model latency yet increase wall time. |
| Qwen improves beyond C-Acc | **Supported** | Embedding-64 improves EtF1, tIoU, tF1@0.3, and tF1@0.5 over controlled 2 FPS; tF1@0.7 is effectively tied. |

## Complete results

Accuracy values are percentages. Lower CardinalityError is better.

### TimeLens2-4B

| Setting | C-Acc ↑ | Card. error ↓ | EtF1 ↑ | tIoU ↑ | tF1@0.3 ↑ | tF1@0.5 ↑ | tF1@0.7 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper-2fps-official | 0.31 | 2.67 | 0.21 | 35.78 | 36.08 | 29.10 | 20.23 |
| paper-2fps-controlled | 24.38 | 1.71 | 17.79 | 50.29 | 53.25 | 43.24 | 29.10 |
| uniform-one-shot-64f | 11.88 | 2.23 | 8.37 | 41.76 | 38.92 | 27.09 | 16.15 |
| full-video-multipass-64f | 8.75 | 2.58 | 4.60 | 37.20 | 28.18 | 15.69 | 7.62 |
| uniform-window-local-64f | 34.06 | 1.52 | 16.40 | 35.23 | 42.73 | 33.22 | 19.72 |
| embedding-window-local-64f | 36.25 | 1.51 | 18.48 | 40.94 | 46.96 | 34.01 | 20.86 |
| uniform-one-shot-128f | 20.00 | 1.93 | 14.46 | 46.74 | 47.21 | 36.95 | 23.19 |
| full-video-multipass-128f | 14.69 | 2.24 | 10.01 | 42.75 | 38.01 | 26.34 | 14.68 |
| uniform-window-local-128f | 33.75 | 1.44 | 17.48 | 38.63 | 48.16 | 37.75 | 23.83 |
| embedding-window-local-128f | 33.12 | 1.48 | 21.14 | 46.58 | 54.93 | 42.72 | 28.48 |

### Qwen3-VL-4B

| Setting | C-Acc ↑ | Card. error ↓ | EtF1 ↑ | tIoU ↑ | tF1@0.3 ↑ | tF1@0.5 ↑ | tF1@0.7 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper-2fps-official | 0.31 | 2.68 | 0.16 | 30.44 | 37.26 | 26.86 | 18.50 |
| paper-2fps-controlled | 2.81 | 3.21 | 2.17 | 30.09 | 32.81 | 27.50 | 20.21 |
| uniform-one-shot-64f | 1.25 | 2.91 | 1.04 | 28.05 | 25.87 | 19.35 | 13.01 |
| embedding-window-local-64f | 22.50 | 1.93 | 13.61 | 37.22 | 41.18 | 31.51 | 20.01 |

## Key paired comparisons

Each cell is `delta [95% CI]`; positive favors the candidate.

| Candidate vs reference | Δ C-Acc | Δ EtF1 | Δ tIoU | Δ F1@0.3 | Δ F1@0.5 | Δ F1@0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| timelens2-4b:controlled-vs-official-2fps | +24.06 [+19.24, +29.07] | +17.58 [+13.79, +21.68] | +14.52 [+11.99, +17.07] | +17.17 [+14.16, +20.22] | +14.14 [+11.04, +17.26] | +8.87 [+6.24, +11.61] |
| timelens2-4b:embedding-window-local-64f-vs-uniform-one-shot-64f | +24.38 [+18.81, +30.15] | +10.11 [+6.79, +13.58] | -0.82 [-2.96, +1.38] | +8.04 [+4.39, +11.69] | +6.92 [+3.23, +10.72] | +4.71 [+1.59, +7.91] |
| timelens2-4b:embedding-window-local-128f-vs-uniform-one-shot-128f | +13.12 [+7.72, +18.54] | +6.68 [+2.82, +10.49] | -0.16 [-2.39, +2.08] | +7.73 [+4.57, +10.93] | +5.77 [+2.09, +9.67] | +5.29 [+2.11, +8.50] |
| timelens2-4b:embedding-window-local-128f-vs-64f | -3.12 [-7.03, +0.91] | +2.66 [+0.26, +5.14] | +5.64 [+4.24, +7.08] | +7.98 [+5.25, +10.78] | +8.71 [+5.80, +11.72] | +7.62 [+4.77, +10.54] |
| timelens2-4b:embedding-64f-vs-controlled-2fps | +11.88 [+5.57, +18.07] | +0.69 [-3.48, +4.76] | -9.36 [-11.87, -6.80] | -6.29 [-10.17, -2.49] | -9.24 [-13.62, -4.93] | -8.24 [-11.95, -4.49] |
| timelens2-4b:embedding-128f-vs-controlled-2fps | +8.75 [+2.82, +14.52] | +3.35 [-0.99, +7.59] | -3.71 [-6.18, -1.22] | +1.68 [-1.59, +5.03] | -0.52 [-4.31, +3.21] | -0.62 [-4.00, +2.83] |
| qwen3-vl-4b:controlled-vs-official-2fps | +2.50 [+0.62, +4.70] | +2.01 [+0.61, +3.69] | -0.35 [-3.14, +2.35] | -4.44 [-7.62, -1.24] | +0.64 [-2.56, +3.90] | +1.71 [-1.14, +4.64] |
| qwen3-vl-4b:embedding-window-local-64f-vs-uniform-one-shot-64f | +21.25 [+16.56, +26.20] | +12.57 [+9.30, +16.22] | +9.17 [+6.65, +11.82] | +15.31 [+11.50, +19.27] | +12.16 [+8.65, +15.76] | +7.00 [+3.87, +10.17] |
| qwen3-vl-4b:embedding-64f-vs-controlled-2fps | +19.69 [+14.66, +25.08] | +11.44 [+7.67, +15.42] | +7.13 [+4.18, +10.06] | +8.37 [+4.32, +12.42] | +4.01 [+0.29, +7.85] | -0.20 [-3.60, +3.28] |
| timelens2-vs-qwen:paper-2fps-controlled | +21.56 [+16.35, +26.90] | +15.62 [+11.43, +19.98] | +20.21 [+17.31, +23.11] | +20.44 [+16.30, +24.36] | +15.74 [+11.72, +19.75] | +8.89 [+5.37, +12.34] |
| timelens2-vs-qwen:uniform-one-shot-64f | +10.62 [+7.21, +14.24] | +7.33 [+4.73, +10.18] | +13.71 [+11.34, +16.16] | +13.05 [+9.05, +16.94] | +7.74 [+4.15, +11.33] | +3.14 [+0.25, +6.10] |
| timelens2-vs-qwen:embedding-window-local-64f | +13.75 [+9.62, +17.96] | +4.87 [+2.66, +7.18] | +3.72 [+2.19, +5.27] | +5.78 [+2.99, +8.62] | +2.50 [-0.38, +5.42] | +0.85 [-1.70, +3.41] |

## Efficiency

| Model | Setting | Input frames | Grounder calls | Sync. model latency (s) | Wall time (s) | Peak VRAM (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| TimeLens2-4B | paper-2fps-official | 141826.00 | 320.00 | 2190.10 | 2190.10 | 10.94 |
| TimeLens2-4B | paper-2fps-controlled | 141826.00 | 320.00 | 2352.60 | 2352.60 | 10.95 |
| Qwen3-VL-4B | paper-2fps-official | 141826.00 | 320.00 | 2223.47 | 2223.47 | 10.93 |
| Qwen3-VL-4B | paper-2fps-controlled | 141826.00 | 320.00 | 2602.41 | 2602.41 | 10.94 |
| TimeLens2-4B | uniform-one-shot-64f | 20480.00 | 320.00 | 300.72 | 427.11 | 9.10 |
| TimeLens2-4B | full-video-multipass-64f | 20480.00 | 640.00 | 405.45 | 532.79 | 8.70 |
| TimeLens2-4B | uniform-window-local-64f | 20480.00 | 701.00 | 345.16 | 604.41 | 9.10 |
| TimeLens2-4B | embedding-window-local-64f | 20628.00 | 701.00 | 386.43 | 658.12 | 9.05 |
| TimeLens2-4B | uniform-one-shot-128f | 40960.00 | 320.00 | 509.11 | 758.39 | 9.90 |
| TimeLens2-4B | full-video-multipass-128f | 40960.00 | 640.00 | 597.35 | 848.42 | 9.10 |
| TimeLens2-4B | uniform-window-local-128f | 40960.00 | 701.00 | 526.04 | 907.75 | 9.90 |
| TimeLens2-4B | embedding-window-local-128f | 40960.00 | 701.00 | 574.95 | 972.21 | 9.85 |
| Qwen3-VL-4B | uniform-one-shot-64f | 20480.00 | 320.00 | 446.82 | 573.63 | 9.10 |
| Qwen3-VL-4B | embedding-window-local-64f | 20628.00 | 701.00 | 367.20 | 639.53 | 9.05 |

Relative to controlled 2 FPS:

- `timelens2-4b:embedding-64f-vs-controlled-2fps`: 85.5% fewer frames, 83.6% less synchronized latency, and 72.0% less wall time.
- `timelens2-4b:embedding-128f-vs-controlled-2fps`: 71.1% fewer frames, 75.6% less synchronized latency, and 58.7% less wall time.
- `qwen3-vl-4b:embedding-64f-vs-controlled-2fps`: 85.5% fewer frames, 85.9% less synchronized latency, and 75.4% less wall time.

The historical `gpu_seconds` field is synchronized end-to-end model-call latency, not pure CUDA kernel time. Runtime values are descriptive because each setting was executed once.

## Prompt audit

- All 320 of 320 official prompts request “the video segment” and “its start and end seconds.”
- All 320 labels contain multiple spans: 2–20, mean 3.67.
- The controlled prompt changes three factors together: plural cardinality instruction, JSON output format, and an explicit duration bound. The current ablation cannot identify which component causes the gain.

| Model | Prompt | Mean predicted spans | Empty | Single | Multiple |
| --- | --- | --- | --- | --- | --- |
| TimeLens2-4B | paper-2fps-official | 1.00 | 1.00 | 318.00 | 1.00 |
| TimeLens2-4B | paper-2fps-controlled | 2.20 | 0.00 | 137.00 | 183.00 |
| Qwen3-VL-4B | paper-2fps-official | 0.98 | 6.00 | 313.00 | 1.00 |
| Qwen3-VL-4B | paper-2fps-controlled | 2.32 | 47.00 | 234.00 | 39.00 |

## Additional findings

- **Naive decomposition hurts.** Full-video multipass is slower and less accurate than uniform one-shot for TimeLens2 at both budgets.
- **Windowing drives cardinality; learned routing refines localization.** Uniform-window-local already captures most of the 64-frame C-Acc gain. At 128 frames, embedding routing adds substantially more EtF1 and tIoU than cardinality accuracy.
- **More frames do not monotonically improve counting.** TimeLens2 embedding-window-local gains localization quality from 64 to 128 frames, while C-Acc decreases slightly.
- **Prompt sensitivity is model-dependent.** The controlled prompt improves all primary TimeLens2 metrics, but on Qwen it improves C-Acc and EtF1 while reducing tF1@0.3 and increasing cardinality error. Prompting is therefore a confound, not a universally effective substitute for model adaptation.
- **Model specialization and inference are complementary in these runs.** Under the same embedding-64 policy, TimeLens2 remains ahead of Qwen in C-Acc (36.25 vs 22.50) and EtF1 (18.48 vs 13.61). At controlled 2 FPS, the gap is also large (17.79 vs 2.17 EtF1).
- Fixed-bin descriptive breakdowns by duration, span count, median span duration, evidence density, and temporal dispersion are available in `omtg_inference_strata.csv`; they are exploratory rather than separately powered hypothesis tests.

## Limitations and reviewer concerns

- The evidence covers one benchmark and two related 4B checkpoints. It is not yet sufficient for a general claim about all multi-span grounding.
- TimeLens2-versus-Qwen is a checkpoint comparison, not a controlled training ablation. The models may differ in data, objectives, and implementation, so their gap cannot be attributed to OMTG training alone.
- The controlled-prompt ablation is bundled; use separate plural-instruction, format, and duration ablations before claiming context engineering is more important than training.
- The 64-frame embedding setting uses 20,628 aggregate frames instead of 20,480, a 148-frame (0.72%) budget overflow. The 128-frame run has no overflow.
- The 2 FPS frame count is duration-derived rather than decoder-measured.
- Router work was executed once and reused across TimeLens2 budgets. Do not sum that cost twice when reconstructing the combined experiment.
- Model loading is excluded, and one run per setting cannot establish runtime variance, energy use, or monetary cost.
- Bootstrap intervals resample videos to preserve dependence among queries sharing a video; they do not quantify uncertainty over random seeds, prompt variants, hardware runs, or model sampling.

## Novelty positioning

The completed v1 is best presented as a controlled study of embedding-routed local inference for frozen set-valued grounders. Hierarchical and training-free selection are established in [DAFS](https://arxiv.org/abs/2607.15689), [SemVID](https://arxiv.org/abs/2603.05663), [TFVTG](https://arxiv.org/abs/2408.16219), and [CoMET-Agent](https://arxiv.org/abs/2606.15320); therefore the report does not claim the first training-free adaptive frame allocator. The proposed v2 contribution is narrower: deterministic residual set search with hard budget accounting and adaptive stopping.
