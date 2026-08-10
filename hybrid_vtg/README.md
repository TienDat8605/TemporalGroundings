# Hybrid VTG

Hybrid VTG is a small, training-free benchmark runner for video temporal grounding. One command selects exactly one method, one frozen model backend, and one official test split. The package intentionally keeps only three experimental methods.

## Core ideas

### `coarse-to-fine-64`

This is the fixed-budget strategy reconstructed from the TimeLens2 `embedding-window-local` experiment. Content changes divide a long video into 20–60 second windows; a Qwen3-VL embedding model ranks those windows against the query; the chosen windows are grounded locally and mapped back to global time. Router frames plus grounder frames never exceed **64**. A short one-window video bypasses routing and gives all 64 frames to the grounder.

### `tpsa-query`

TPSA-query means **query-aware evidence selection after the video encoder**. It encodes a uniform full-timeline sample, scores every encoded visual unit against the text query, and retains an exact 12.5% evidence budget by default. The selection combines high-relevance units with deterministic endpoint and temporal-band anchors, so a strong local match cannot erase the rest of the timeline. Selected evidence stays in chronological order and is passed to one final prediction call.

### `hmve`

HMVE means **hierarchical multi-view evidence**. A low-rate full-video scout first preserves global coverage and proposes up to four query-relevant temporal corridors. Those corridors are encoded in more detail. Scout and detail evidence are then merged by absolute timestamp, near-duplicates are removed, one scout anchor per temporal location is protected, and the compact pack is used for one final prediction. HMVE changes what the model observes; it is not another TPSA scoring variant.

## SemVID relationship

SemVID is no longer a runtime dependency, submodule, baseline, selector, prompt template, or policy source. Hybrid VTG keeps only one low-level implementation lesson from it: when already-encoded Qwen visual embeddings are inserted into a compact multimodal prefill, explicit first-step `position_ids` must survive `prepare_inputs_for_generation`. This is needed because TPSA-query and HMVE pass selected encoder outputs rather than asking the standard processor to encode an untouched video prompt. The three selection methods were reimplemented around the contracts in this repository and do not call SemVID code.

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Raw SlowFast extraction for a SlowFast + CLIP UniVTG checkpoint is optional:

```bash
pip install -e '.[univtg-video]'
```

CUDA is strongly recommended for the 4B generative models. Model weights are downloaded by Transformers unless `--checkpoint` names a local directory.

## One run interface

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data /datasets/omtg \
  --model qwen3-vl-4b \
  --method coarse-to-fine-64 \
  --subset 10 \
  --seed 42
```

The only subset choices are `10`, `20`, and `100`. Sampling is over queries, not videos: IDs are sorted, shuffled with the supplied seed, then the first `ceil(N × percentage / 100)` queries are used. Therefore 10% is a prefix of 20%, and 20% is a prefix of 100% for the same dataset and seed. Re-running a larger percentage resumes the existing run by sample ID.

### Benchmarks

Only official test annotations are loaded:

| CLI name | Required annotation name | Video location |
|---|---|---|
| `omtg` | `OMTGBench.tsv` | Any subdirectory below `--data` |
| `tacos` | `test.jsonl`, `annotations/test.jsonl`, or `captions/test.jsonl` | Any subdirectory below `--data` |
| `qvhighlights` | `highlight_test_release.jsonl` in the root, `annotations/`, or `metadata/` | Any subdirectory below `--data` |

Video files are discovered recursively by stem. OMTG is evaluated as multi-interval grounding. TACoS reports single-result moment-retrieval metrics against all reference windows. Official QVHighlights test labels are hidden, so a 100% run creates a moment-retrieval submission without local metrics or saliency predictions.

### Models

| CLI name | Default checkpoint | Notes |
|---|---|---|
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | Direct Transformers adapter; no SemVID checkout |
| `timelens2-4b` | `MCG-NJU/TimeLens2-4B` | Uses the same evidence interface; see its restrictive research license |
| `univtg` | none | Pass an official moment-retrieval checkpoint with `--checkpoint` |

UniVTG checkpoint shapes configure the inference network, so pretraining-only, omnibus, and downstream moment-retrieval checkpoints are accepted when their feature stack is supported. Choose `--model-spec clip-b16`, `clip-b32`, or `slowfast-clip-b32` if checkpoint metadata is missing.

Raw extraction and the repository cache work without extra arguments. To use official `.npz` features, pass one directory per feature stream; files must be named `<video-id>.npz` with a `features` array. Streams are concatenated in argument order:

```bash
hybrid-vtg run \
  --benchmark tacos \
  --data /datasets/tacos \
  --model univtg \
  --checkpoint /checkpoints/univtg_tacos/model_best.ckpt \
  --model-spec slowfast-clip-b32 \
  --feature-root /features/tacos/slowfast \
  --feature-root /features/tacos/clip \
  --method hmve \
  --subset 20 \
  --seed 42
```

## Results

All outputs live under one directory:

```text
results/
├── RESULTS.md                 generated summary table
├── index.csv                  generated machine-readable index
├── cache/                     shared decoded-frame and feature cache
├── runs/<benchmark>/<model>/<method>/seed-<seed>/
│   ├── manifest.json          immutable run identity and shuffled ID order
│   ├── predictions.jsonl      append-only successful predictions
│   ├── errors.jsonl           per-query failures
│   ├── cache/                 method-local intermediates
│   └── metrics/p010|p020|p100.json
├── submissions/<benchmark>/<model>/<method>/seed-<seed>.jsonl
└── legacy/                    curated pre-refactor evidence
```

`RESULTS.md` and `index.csv` are rebuilt after every run. A manifest mismatch is rejected rather than mixing configurations. Inference is batch size one.

## Add another method, model, or benchmark

The stable contracts are in `src/hybrid_vtg/contracts.py`:

- A method implements `Method.run(sample, model, cache_dir)` and lives in its own folder under `methods/`.
- A model implements `ModelBackend.encode`, `query_scores`, and `predict` under `models/`.
- A benchmark implements `Benchmark.load_test` and `evaluate` under `benchmarks/`.
- Each package adds its factory to its explicit `register_*` function. There is no import-time plugin discovery or model-specific branch in the runner.

Run the checks with:

```bash
ruff check src tests
pytest
```

## Provenance and licenses

The trimmed UniVTG transformer and positional-encoding implementation is derived from the official [showlab/UniVTG](https://github.com/showlab/UniVTG) repository under MIT. The fixed-budget method is a clean reimplementation of the TimeLens2 evaluation policy. Model checkpoints and datasets are not redistributed here and retain their own terms. Read [NOTICE.md](NOTICE.md) and [LICENSES](LICENSES/) before use; in particular, TimeLens2 is academic-only and states that it is not intended for use within the European Union.
