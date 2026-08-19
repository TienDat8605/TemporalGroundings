# Idea 3: Scout-Guided Dense Evidence Grounding (SGDE)

This directory contains research notes, architectural designs, and benchmark reports for **Idea 3**:

- **[`adaptive_sgde_pipeline.md`](./adaptive_sgde_pipeline.md)**: Full mathematical formulation, adaptive corridor planning algorithm, calibrated vision processing pipeline, and benchmark results against Native TimeLens2-4B.
- **[`new_idea.md`](./new_idea.md)**: Original conceptual proposal and design rationale for decoupling global scouting from local dense evidence verification.

## Current Benchmark Highlights (QVHighlights-TimeLens)

- **Adaptive SGDE**: **58.42% mIoU**, **76.77% R@1(0.3)** (~2.2s/sample)
- **Native TimeLens2-4B**: 57.66% mIoU, 74.19% R@1(0.3) (~4.55s/sample)
