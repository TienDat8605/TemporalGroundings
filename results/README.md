# OMTG experiment results

This is the single location for experiment outputs, invocation logs, and
comparison notes. The runnable implementation is under `../TimeLens2`; the
official OMTG repository is pinned at `../TimeLens2/OMTG`.
The reference revision is `3291c9e19490db19c7aae593791f80aaca6f53a8`.

## Layout

- `omtg_2fps/`: paper-style 2 FPS runs for TimeLens2-4B and Qwen3-VL-4B.
- `omtg_fixed_frame/`: fixed-budget hierarchical and uniform runs.
- `omtg_residual_search/`: destination for strictly budgeted residual-search runs.
- `colab_runs/`: preserved remote job specifications, logs, status, and checkpoints.
- `comparison.md`: research assessment of all 14 complete TimeLens2 and
  Qwen3-VL settings, including video-clustered paired bootstrap intervals.
- `omtg_inference_comparison.json`, `omtg_inference_results.csv`, and
  `omtg_inference_deltas.csv`: machine-readable comparison outputs.
- `omtg_inference_strata.csv`: descriptive results split by duration, event
  count, span duration, evidence density, and temporal dispersion.

Each run directory keeps its configuration, predictions, summaries, and
append-only `invocations.jsonl` log together.

## Main findings

- Inference policy materially affects multi-span grounding. On base
  Qwen3-VL-4B, embedding-window-local at 64 frames raises C-Acc from 1.25 to
  22.50 and EtF1 from 1.04 to 13.61 relative to uniform one-shot.
- Model specialization and inference appear complementary. Under the same embedding
  policy, TimeLens2-4B still leads Qwen3-VL-4B by 13.75 C-Acc points and 4.87
  EtF1 points.
- TimeLens2 embedding-64 improves cardinality over controlled 2 FPS but loses
  localization accuracy. Embedding-128 approximately matches thresholded
  localization, improves C-Acc by 8.75 points, and has an observed EtF1 gain
  of 3.35 points whose confidence interval includes zero.
- Every official prompt asks for a singular segment even though every label
  contains multiple spans. This instruction–annotation mismatch is a
  benchmark confound; the controlled ablation does not isolate instruction
  plurality, output format, and duration context.
- Fixed-frame embedding inference uses 59–75% less wall time than controlled
  2 FPS in these runs. This does not mean it is always faster than fixed-frame
  uniform inference.

Timing fields named `gpu_seconds` in historical raw summaries are synchronized
end-to-end model-call latency, not pure CUDA kernel time. See `comparison.md`
for the complete tables, confidence intervals, accounting notes, and reviewer
limitations.
