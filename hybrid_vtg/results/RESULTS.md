# Hybrid VTG results

This file is generated from run manifests and metrics. QVHighlights test runs are
submission-only because the official labels are hidden.

| Benchmark | Model | Method | Seed | Subset | Success | Metrics |
|---|---|---|---:|---:|---:|---|
| qvhighlights-timelens | timelens2-4b | anchored-corridor-64 | 42 | 010% | 155/155 | `{"R@1,IoU=0.3": 0.2645161290322581, "R@1,IoU=0.5": 0.13548387096774195, "R@1,IoU=0.7": 0.03870967741935484, "count": 155, "mIoU": 0.1702434884447061}` |
| qvhighlights-timelens | timelens2-4b | coarse-to-fine-64 | 42 | 010% | 154/155 | `{"R@1,IoU=0.3": 0.14193548387096774, "R@1,IoU=0.5": 0.04516129032258064, "R@1,IoU=0.7": 0.012903225806451613, "count": 155, "mIoU": 0.1126326928279243}` |

Historical artifacts retained during the refactor are under `legacy/`; their
completeness and provenance are documented in `legacy/README.md`.
