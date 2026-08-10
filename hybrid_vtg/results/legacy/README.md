# Curated historical results

These artifacts predate the current contracts and are evidence, not resumable new-schema runs. They were selected from the former top-level `results/` directory on 2026-08-10. Metrics below were either stored with the run or recomputed from its saved intervals using the current metric functions; percentages are shown as percentages.

## Result summary

| Method / model | Benchmark | Coverage | Main result | Status |
|---|---|---:|---|---|
| Coarse-to-fine / Qwen3-VL-4B | OMTG | 320/320 | 64f embedding: C-Acc 22.50, EtF1 13.61, tIoU 37.22 | Complete historical run |
| Coarse-to-fine / TimeLens2-4B | OMTG | 320/320 | 64f embedding: C-Acc 36.25, EtF1 18.48, tIoU 40.94 | Complete historical run |
| Coarse-to-fine / TimeLens2-4B | OMTG | 320/320 | 128f embedding: C-Acc 33.12, EtF1 21.14, tIoU 46.58 | Complete historical run; 128f is outside the retained 64f method |
| TPSA-query / Qwen Thinking | OMTG diagnostic | 64/64 | C-Acc 50.00, EtF1 38.59, tIoU 57.89 | Complete diagnostic subset, not seed-sampled |
| TPSA-query / Qwen Thinking | OMTG | 320/320 | C-Acc 0.94, EtF1 0.60, tIoU 3.45 | Complete early run; clear regression, retained as negative evidence |
| TPSA-query / Qwen Thinking | TACoS | 1,664/1,664 | mIoU 16.68, R@1 IoU .3/.5/.7 = 23.80/15.50/6.19 | Complete for its historical annotation file |
| HMVE / Qwen Thinking | OMTG | 272/320 | C-Acc 24.63, EtF1 16.69, tIoU 42.03 | Incomplete; do not compare directly with full runs |

The fixed-frame study also contains uniform and other historical controls. Its strongest controlled finding was that embedding-window-local at 64 frames materially improved Qwen over uniform-one-shot (C-Acc 22.50 vs 1.25; EtF1 13.61 vs 1.04). TimeLens2 improved from C-Acc 11.88 / EtF1 8.37 under uniform 64 frames to 36.25 / 18.48 under embedding-window-local. The detailed paired-bootstrap analysis is in `reports/omtg_inference_comparison.md`.

## Important limitations

- The old 64-frame embedding runs counted 20,628 aggregate frames instead of 20,480: a 148-frame, 0.72% overflow. The current `coarse-to-fine-64` implementation enforces the budget and is therefore not byte-for-byte equivalent.
- Historical TPSA/HMVE runs used `Qwen/Qwen3-VL-4B-Thinking` and a SemVID-based runtime. The current implementation uses Qwen3-VL-4B-Instruct by default and has no SemVID runtime. Treat these as prior evidence, not current-method scores.
- The 64-query TPSA diagnostic is the first 64 historical IDs, not the new seeded uniform subset.
- The full early OMTG TPSA run is inconsistent with its later diagnostic and is retained specifically to expose that failed configuration.
- HMVE stopped at 272 queries, so its recomputed metrics are descriptive only.
- Timing fields called `gpu_seconds` in fixed-frame summaries are synchronized end-to-end model-call latency, not kernel time.

## Layout and integrity

```text
legacy/
├── coarse_to_fine_64/omtg-study/      full fixed-frame predictions and summaries
├── tpsa_query/omtg/diagnostic-64/     predictions, manifest, stored metrics
├── tpsa_query/omtg/full-320/           negative early full run
├── tpsa_query/tacos/full-1664/         predictions and manifest
├── hmve/omtg/partial-272/              predictions and manifest
└── reports/                            aggregate tables, bootstrap outputs, old overview
```

SHA-256 checksums for the principal raw files:

```text
461a2d409b6b7afa497fcbb18313393fad64b1215da0713e68d6b090e2212893  tpsa_query/omtg/full-320/predictions.jsonl
bbeac86022ab728f1995c06b62d90a3cd4fd0a7a0fa0a0586c174e204438c3ca  tpsa_query/omtg/diagnostic-64/predictions.jsonl
9c1b2e66fc0ccbee8cc68dae81db401bdccbb7a10cccf7811373d2c9d60efbf0  tpsa_query/tacos/full-1664/predictions.jsonl
38d181fa0cf48be9c2c17d697d79e9b2af6b15cd8586a78f02c2149607241e39  hmve/omtg/partial-272/predictions.jsonl
```

Removed during curation: Colab job logs/checkpoint tarballs, raw 2-FPS predictions already represented in the aggregate report, superseded TPSA-motion/TPSA-boundary experiments, a duplicate incomplete 1,168-row TACoS file, and abandoned residual-search implementation artifacts. The residual-search negative-result report remains because it records the scientific conclusion without retaining that method in the codebase.
