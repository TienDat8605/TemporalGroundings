# Hybrid VTG

Hybrid VTG is a small, training-free benchmark runner for video temporal grounding. One command selects exactly one method, one frozen model backend, and one official test split. The package intentionally keeps only three experimental methods.

## Core ideas

### `coarse-to-fine-64`

This is the fixed-budget strategy reconstructed from the TimeLens2 `embedding-window-local` experiment. Content changes divide a long video into 20–60 second windows; a Qwen3-VL embedding model ranks those windows against the query; the chosen windows are grounded locally and mapped back to global time. Router frames plus grounder frames never exceed **64**. A short one-window video bypasses routing and gives all 64 frames to the grounder.

### `tpsa-query`

TPSA-query means **query-aware evidence selection after the video encoder**. It encodes a uniform full-timeline sample, scores every encoded visual unit against the text query, and retains an exact 12.5% evidence budget by default. The selection combines high-relevance units with deterministic endpoint and temporal-band anchors, so a strong local match cannot erase the rest of the timeline. Selected evidence stays in chronological order and is passed to one final prediction call.

### `hmve`

HMVE means **hierarchical multi-view evidence**. It makes exactly three visual observation passes: a 0.5 FPS full-video scout, 1 FPS query-relevant corridor refinement, and the densest 2 FPS observation around relevance rises and falls that may indicate event boundaries. Each pass is batched into one encoder call. Evidence from all three passes is merged by absolute timestamp, near-duplicates are removed, one real scout anchor per temporal location is protected, and the compact pack is used for exactly one final prediction. Spatial crops and in-encoder pruning remain future extensions.

## SemVID relationship

SemVID is no longer a runtime dependency, submodule, baseline, selector, prompt template, or policy source. Hybrid VTG keeps only one low-level implementation lesson from it: when already-encoded Qwen visual embeddings are inserted into a compact multimodal prefill, explicit first-step `position_ids` must survive `prepare_inputs_for_generation`. This is needed because TPSA-query and HMVE pass selected encoder outputs rather than asking the standard processor to encode an untouched video prompt. The three selection methods were reimplemented around the contracts in this repository and do not call SemVID code.

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[downloads,test]'
```

Video decoding and PySceneDetect use the headless OpenCV wheel and do not need
`libGL.so.1`. If a GUI OpenCV wheel was previously installed in the environment,
remove the conflict once and reinstall the project:

```bash
pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless
pip install --force-reinstall 'opencv-python-headless>=4.9,<5'
pip install -e '.[downloads,test]'
```

Raw SlowFast extraction for a SlowFast + CLIP UniVTG checkpoint is optional:

```bash
pip install -e '.[univtg-video]'
```

CUDA is strongly recommended for the 4B generative models. Model weights are downloaded by Transformers unless `--checkpoint` names a local directory.

## Download datasets and checkpoints

One command downloads the three benchmarks, TimeLens2-4B, and the official
UniVTG CLIP-B/32 4M pretraining checkpoint:

```bash
hybrid-vtg download --root ./assets --accept-licenses
```

Downloads now show byte progress, transfer speed, ETA, extraction progress,
and overall asset progress. Interrupted HTTP transfers resume from their
existing `.part` file.

For authenticated Hugging Face downloads, add `--hf-login`:

```bash
hybrid-vtg download --root ./assets --accept-licenses --hf-login
```

This uses `HF_TOKEN` when that environment variable is set; otherwise it opens
Hugging Face's secure token prompt and saves the login in the standard local
Hugging Face token store. Authentication can avoid anonymous rate limits for
OMTG, TACoS, QVHighlights, and TimeLens2 downloads. The token is never stored
in `assets/manifest.json`.

The equivalent repository script is:

```bash
python scripts/download_assets.py --root ./assets --accept-licenses
```

To download only selected assets, name them explicitly:

```bash
hybrid-vtg download omtg qvhighlights timelens2-4b --root ./assets --accept-licenses
```

Valid targets are `omtg`, `tacos`, `qvhighlights`, `timelens2-4b`, and
`univtg`. With no targets, all five are downloaded. The layout is:

```text
assets/
├── manifest.json
├── datasets/
│   ├── omtg/             OMTGBench.tsv + videos/
│   ├── tacos/            test annotations + 25 compressed test videos
│   └── qvhighlights/     test annotation + its 1,529 test videos only
└── checkpoints/
    ├── timelens2-4b/
    └── univtg-pretrained-clip-b32-4m/
```

HTTP downloads resume from `.part` files. Source archives are removed only
after successful extraction to avoid retaining duplicate copies. Every
completed target contains `.complete.json`; the UniVTG marker lists the exact
downloaded `.ckpt` path to pass to `--checkpoint`.

`--accept-licenses` confirms that you reviewed the upstream terms. TACoS uses
VideoMind's compressed 3 FPS, 480p, no-audio release and matching test
annotations; the archive is about 1.49 GB rather than the 30.2 GB original-video
archive. The archive contains all splits, so the downloader selectively extracts
and retains only the 25 test videos. The VideoMind repository declares
BSD-3-Clause, while the underlying
TACoS data retains its upstream terms. QVHighlights annotations use CC BY-NC-SA
4.0. The script does not mirror or redistribute any dataset. Sources are the
[OMTG Bench release](https://huggingface.co/datasets/insomnia7/omtg_bench),
[QVHighlights/Moment-DETR](https://github.com/jayleicn/moment_detr),
[QVHighlights test archive](https://huggingface.co/datasets/jwnt4/qvhighlights-test),
[VideoMind TACoS](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos),
[TACoS](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus),
[TimeLens2](https://huggingface.co/MCG-NJU/TimeLens2-4B), and
[UniVTG](https://github.com/showlab/UniVTG).

## One run interface

```bash
hybrid-vtg run \
  --benchmark omtg \
  --model qwen3-vl-4b \
  --method coarse-to-fine-64 \
  --subset 10 \
  --seed 42
```

When `--data` is omitted, it defaults to
`./assets/datasets/<benchmark>`, matching `hybrid-vtg download --root ./assets`.
Use `--data` only when the assets live elsewhere.

`--subset` accepts any percentage from `0` through `100`, including decimals such as `12.5`. Sampling is over queries, not videos: IDs are sorted, shuffled with the supplied seed, then the first `ceil(N × percentage / 100)` queries are used. For the same dataset and seed, every smaller percentage is a prefix of every larger percentage. Re-running a larger percentage resumes the existing run by sample ID. A 0% run validates the dataset and writes empty metrics without loading a model.

### Benchmarks

Only official test annotations are loaded:

| CLI name | Required annotation name | Video location |
|---|---|---|
| `omtg` | `OMTGBench.tsv` | Any subdirectory below `--data` |
| `tacos` | `test.jsonl`, `annotations/test.jsonl`, or `captions/test.jsonl` | Any subdirectory below `--data` |
| `qvhighlights` | `highlight_test_release.jsonl` in the root, `annotations/`, or `metadata/` | Any subdirectory below `--data` |

Video files are discovered recursively by stem. OMTG is evaluated as multi-interval grounding. TACoS reports single-result moment-retrieval metrics against all reference windows. Official QVHighlights test labels are hidden, so a 100% run creates a moment-retrieval submission without local metrics or saliency predictions.

Prepared benchmark payloads are test-only: OMTG contains 320 test queries over
287 benchmark videos; TACoS contains 4,001 test queries over 25 videos; and
QVHighlights contains 1,542 test queries over 1,529 videos. The TACoS source
archive is all-split, but non-test members are never extracted on a fresh run
and are pruned from older completed downloads.

The QVHighlights downloader uses one resumable test-only archive containing
exactly the 1,529 unique video IDs in the official test annotation. It is about
17.6 GB, does not download train or validation videos, and avoids one API
request per video. The old all-splits archive was about 134 GB.

### Models

| CLI name | Default checkpoint | Notes |
|---|---|---|
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | Direct Transformers adapter; no SemVID checkout |
| `timelens2-4b` | `MCG-NJU/TimeLens2-4B` | Official Apache-2.0 checkpoint; local downloader path is `assets/checkpoints/timelens2-4b` |
| `univtg` | none | Pass a `.ckpt` file or a directory containing exactly one `.ckpt`; the downloader directory works directly |

UniVTG checkpoint shapes configure the inference network, so pretraining-only, omnibus, and downstream moment-retrieval checkpoints are accepted when their feature stack is supported. Choose `--model-spec clip-b16`, `clip-b32`, or `slowfast-clip-b32` if checkpoint metadata is missing.

Raw extraction and the repository cache work without extra arguments. To use official `.npz` features, pass one directory per feature stream; files must be named `<video-id>.npz` with a `features` array. Streams are concatenated in argument order:

The downloaded pretraining-only UniVTG checkpoint can be passed directly as
`--checkpoint assets/checkpoints/univtg-pretrained-clip-b32-4m` with
`--model-spec clip-b32`.

```bash
hybrid-vtg run \
  --benchmark tacos \
  --model univtg \
  --checkpoint assets/checkpoints/univtg-pretrained-clip-b32-4m \
  --model-spec clip-b32 \
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
│   └── metrics/p<percentage>.json
├── submissions/<benchmark>/<model>/<method>/seed-<seed>.jsonl
└── legacy/                    curated pre-refactor evidence
```

`RESULTS.md` and `index.csv` are rebuilt after every run. A manifest mismatch is rejected rather than mixing configurations. Inference is batch size one.
The command prints the absolute run directory before model loading and shows a
per-sample progress bar. If every sample fails, metrics are left as `null`
instead of reporting failures as model predictions; inspect `errors.jsonl` for
the underlying exceptions.

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
