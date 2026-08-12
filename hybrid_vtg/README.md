# Hybrid VTG

Hybrid VTG is a small benchmark runner for frozen-model video temporal grounding. One command selects exactly one method, one frozen model backend, and one official test split. Most methods are fully training-free; UniTime experiments use its already-trained public adapter without any additional optimization.

## Core ideas

### `coarse-to-fine-64`

This is the fixed-budget strategy reconstructed from the TimeLens2 `embedding-window-local` experiment. Content changes divide a long video into 20–60 second windows; a Qwen3-VL embedding model ranks those windows against the query; the chosen windows are grounded locally and mapped back to global time. Router frames plus grounder frames never exceed **64**. A short one-window video bypasses routing and gives all 64 frames to the grounder.

### `hmve`

HMVE means **hierarchical multi-view evidence**. It makes exactly three visual observation passes: a 0.5 FPS full-video scout, 1 FPS query-relevant corridor refinement, and the densest 3 FPS observation around relevance rises and falls that may indicate event boundaries. Each pass is batched into one encoder call. Evidence from all three passes is merged by absolute timestamp, near-duplicates are removed, one real scout anchor per temporal location is protected, and the compact pack is used for exactly one final prediction.

### `unitime-fixed` and `unitime-adaptive`

These methods use the public UniTime LoRA on its original frozen Qwen2-VL-7B base. `unitime-fixed` is a clean-room structural baseline: videos longer than 64 seconds receive one trained coarse timestamp-retrieval call over fixed 32-second groups, followed by one fine grounding call. It preserves UniTime's timestamp-interleaved hierarchy but uses this project's processor-compatible, grid-aligned frame scaling instead of upstream long-video post-encoder interpolation. `unitime-adaptive` replaces the fixed single corridor with a training-free HMVE scout, top-k query-relevant corridors, and high-rate boundary observations, then performs one timestamp-interleaved UniTime generation. `--corridor-top-k` accepts one through eight retained corridors and defaults to four.

UniTime itself is trained. These integrations are **post-hoc training-free**: the released adapter and base model remain frozen and this project performs no additional optimization. The fixed method is the controlled dense baseline for evaluating the adaptive method; use the upstream implementation when an exact reproduction of published UniTime numbers is required.

## Independent visual-token pruning

Qwen and TimeLens2 expose two separate, optional pruning points. Both are disabled by default, so existing runs keep their original behavior.

- `mage` is an **encoder-stage** policy. It computes camera-compensated optical flow and a motion-compensated luminance residual from the decoded frames, resizes that importance map to Qwen's processed patch grid, and keeps complete `spatial_merge_size²` patch groups before a configurable vision-transformer block. Periodic dense temporal anchors preserve context; non-anchor cells compete globally by motion/residual importance. The default layer `0` therefore skips all vision-transformer blocks for removed patches, though patch embedding itself remains dense.
- `semvid` is a **post-encoder** policy adapted from the official Apache-2.0 SemVID Qwen3-VL selector. It allocates a query-and-motion-weighted budget over timestamps, then keeps context prototypes, query-aware diverse object tokens, and motion tokens. It runs immediately before compact language-model prefill.

The Mage policy imports Mage-VL's dense-anchor/sparse-update idea, but it is not the paper's trained Mage-ViT and currently uses decoded-frame optical flow/residual maps rather than exported codec motion vectors. This distinction matters: Qwen was not trained on Mage-ViT's sparse codec-token format. The implementation instead retains complete Qwen merger cells and their original rotary coordinates.

Retention values are fractions of the original dense Qwen evidence count. At least one cell per temporal unit is always kept for coherence, so extremely small ratios can be raised to that floor. When both policies are enabled, `--post-retention` must not exceed `--encoder-retention`.

Example: keep 50% of merger cells before vision block 0, then let SemVID reduce the final evidence to 12.5% of the original dense count:

```bash
hybrid-vtg run \
  --benchmark omtg \
  --model qwen3-vl-4b \
  --method hmve \
  --subset 10 \
  --seed 42 \
  --encoder-pruning mage \
  --encoder-retention 0.5 \
  --encoder-prune-layer 0 \
  --post-pruning semvid \
  --post-retention 0.125
```

The same policies work independently with frozen UniTime. See
[Run UniTime experiments](#run-unitime-experiments) for the complete baseline,
Mage, SemVID, and adaptive command matrix.

All Qwen-based backends use an explicit 4,096-cell visual-evidence budget. The UniTime and base Qwen2 backends also cache grid-aligned post-encoder features by video identity, timestamps, checkpoints, pruning configuration, and evidence budget. Adaptive routing uses 0.5 FPS scouting, 1 FPS corridor refinement, and 2 FPS boundary refinement. Sparse MRoPE coordinates are retained and generated boundaries are snapped to timestamps shown in the prompt. For `--model unitime`, `--checkpoint` overrides the default `zeqianli/UniTime` adapter and `--base-checkpoint` overrides `Qwen/Qwen2-VL-7B-Instruct`.

TimeLens2-4B and TimeLens-7B expose two deliberately separate paths.
`timelens-native` uses each checkpoint's official whole-video 2 FPS prompt with
the shared 4,096-token budget; it is the dense baseline and rejects pruning flags.
`unitime-adaptive` uses this project's timestamp-interleaved evidence interface
and supports Mage and SemVID. The shared ceiling keeps long TACoS videos
practical on a 24 GB GPU and applies to base Qwen3-VL as well.

Use either policy alone by omitting the other policy's two arguments. The old generic `--prune-ratio` and `--prune-layer` interface has been removed. Pruned configurations receive distinct result directories such as `qwen3-vl-4b--enc-mage-r0.5-l0--post-semvid-r0.125`, preventing them from being mixed with dense baselines.

## Install

Python 3.10 or newer is required. For GPU runs, install the PyTorch build that
matches the machine's CUDA driver first, following the PyTorch installation
selector. Then, from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[downloads,test]'
```

The base installation already contains everything needed by Qwen2-VL,
Qwen2.5-VL, Qwen3-VL, UniTime, TimeLens, and TimeLens2; there are no separate
model requirements files:

| Dependency | Purpose |
|---|---|
| `torch>=2.4`, `torchvision>=0.19` | Model execution and visual preprocessing |
| `transformers>=4.57,<5` | Qwen2-VL/Qwen2.5-VL/Qwen3-VL models, processors, and generation |
| `peft>=0.18,<1` | Loading the released UniTime LoRA adapter without training |
| `accelerate>=1.0` | Automatic device placement for the 4B/7B backends |
| `Pillow`, `opencv-python-headless`, `numpy` | Frame decoding, resizing, motion, and residual maps |
| `qwen-vl-utils[decord]>=0.0.14` | Official whole-video TimeLens/TimeLens2 preprocessing and video decoding |

`.[downloads]` adds `huggingface-hub` and `gdown`; `.[test]` adds `pytest` and
`ruff`; `.[univtg-video]` adds PyTorchVideo only for raw SlowFast extraction.
The dependency resolver has been checked against the declared ranges. A fresh
virtual environment is recommended because unrelated scientific packages in a
shared environment can impose conflicting NumPy, protobuf, or OpenCV pins.

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

CUDA is strongly recommended for the 4B and 7B generative models. UniTime loads
both `Qwen/Qwen2-VL-7B-Instruct` and the `zeqianli/UniTime` PEFT adapter; model
weights are downloaded from Hugging Face unless the checkpoint flags name local
directories. Use `--base-checkpoint` for the Qwen2-VL base and `--checkpoint`
for the adapter.

FlashAttention 2 is optional, not required for correctness. If it is installed
and compatible with the active CUDA/PyTorch build, the backends select it
automatically. It is strongly recommended for the official whole-video
TimeLens controls on a 24 GB GPU; follow the model card's installation command
instead of adding it to the portable base requirements.

## Run TACoS: UniTime, TimeLens2, and TimeLens

Download the TACoS test split and all three model releases once:

```bash
hybrid-vtg download tacos unitime timelens2-4b timelens-7b \
  --root ./assets --accept-licenses --hf-login
```

Start with a reproducible 10% subset; change `--subset 10` to `--subset 100`
for the full 4,001-query TACoS test split.

To run the complete nine-run 10% matrix in a detached tmux session, use:

```bash
scripts/run_tacos_tmux.sh
tmux attach -t tacos-vtg-10
```

The detached worker changes to the repository root and activates `.venv`
itself, so it keeps the correct environment after SSH disconnects. It runs the
three original/native controls, three adaptive top-k=4 experiments, and three
adaptive + Mage (50%) + SemVID (12.5%) experiments sequentially on GPU 0.
Matching partial benchmark outputs resume normally; one failed run is logged
without stopping the remaining matrix. Logs are written under
`results/logs/tacos-matrix/`. Override defaults with environment variables:

```bash
TACOS_GPU=1 TACOS_TMUX_SESSION=tacos-gpu1 scripts/run_tacos_tmux.sh
```

### Dense fixed-budget controls

```bash
# UniTime's fixed coarse-to-fine hierarchy
hybrid-vtg run --benchmark tacos --model unitime --method unitime-fixed \
  --checkpoint ./assets/checkpoints/unitime/adapter \
  --base-checkpoint ./assets/checkpoints/unitime/qwen2-vl-7b \
  --subset 10 --seed 42

# TimeLens2-4B's native whole-video 2 FPS prompt, capped at 4,096 tokens
hybrid-vtg run --benchmark tacos --model timelens2-4b --method timelens-native \
  --checkpoint ./assets/checkpoints/timelens2-4b \
  --subset 10 --seed 42

# TimeLens-7B's official interleaved-text-timestamp inference
hybrid-vtg run --benchmark tacos --model timelens-7b --method timelens-native \
  --checkpoint ./assets/checkpoints/timelens-7b \
  --subset 10 --seed 42
```

### Our adaptive temporal method

Run the same training-free scout/corridor/boundary method on each checkpoint:

```bash
hybrid-vtg run --benchmark tacos --model unitime --method unitime-adaptive \
  --checkpoint ./assets/checkpoints/unitime/adapter \
  --base-checkpoint ./assets/checkpoints/unitime/qwen2-vl-7b \
  --corridor-top-k 4 --subset 10 --seed 42

hybrid-vtg run --benchmark tacos --model timelens2-4b --method unitime-adaptive \
  --checkpoint ./assets/checkpoints/timelens2-4b \
  --corridor-top-k 4 --subset 10 --seed 42

hybrid-vtg run --benchmark tacos --model timelens-7b --method unitime-adaptive \
  --checkpoint ./assets/checkpoints/timelens-7b \
  --corridor-top-k 4 --subset 10 --seed 42
```

For each adaptive command, append exactly one of these blocks to measure the
spatial method independently:

```bash
# Mage: retain 50% of dense merger cells before vision block 0
--encoder-pruning mage --encoder-retention 0.5 --encoder-prune-layer 0

# SemVID: retain 12.5% of original dense evidence after the encoder
--post-pruning semvid --post-retention 0.125
```

You can also combine both blocks; `0.125` remains relative to the original
dense evidence, so it is valid with Mage retention `0.5`. A focused matrix has
four configurations per model: dense control, adaptive, adaptive + Mage, and
adaptive + SemVID. UniTime's dense control is `unitime-fixed`; both TimeLens
dense controls are `timelens-native`. Do not attach pruning flags to
`timelens-native`, because that would no longer be the dense native control.

## Run UniTime experiments

The CLI separates the model from the inference method:

- `--model unitime` loads frozen Qwen2-VL-7B plus the released UniTime LoRA.
- `--model qwen2-vl-7b` uses the frozen Qwen2-VL base without the adapter.
- `--model qwen3-vl-4b` uses frozen Qwen3-VL with its native evidence prompt.
- `--method unitime-fixed` uses fixed 32-second coarse segments.
- `--method unitime-adaptive` replaces fixed coarse retrieval with HMVE top-k corridors.
- Mage and SemVID are independent model options. Adaptive routing supports all
  three Qwen configurations; fixed coarse retrieval is a Qwen2/UniTime path.

Prepare OMTG and authenticate with Hugging Face if needed:

```bash
hybrid-vtg download omtg --root ./assets --accept-licenses --hf-login
```

The examples below use 10% of OMTG with seed 42. Change `--subset 10` to
`--subset 100` for the complete test split.

### 1. UniTime adaptive, no spatial pruning

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model unitime \
  --method unitime-adaptive \
  --corridor-top-k 4 \
  --subset 10 \
  --seed 42
```

### 2. Fixed UniTime with Mage encoder pruning

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model unitime \
  --method unitime-fixed \
  --subset 10 \
  --seed 42 \
  --encoder-pruning mage \
  --encoder-retention 0.5 \
  --encoder-prune-layer 0
```

### 3. Fixed UniTime with SemVID post-encoder pruning

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model unitime \
  --method unitime-fixed \
  --subset 10 \
  --seed 42 \
  --post-pruning semvid \
  --post-retention 0.125
```

### 4. UniTime adaptive with Mage encoder pruning

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model unitime \
  --method unitime-adaptive \
  --corridor-top-k 4 \
  --subset 10 \
  --seed 42 \
  --encoder-pruning mage \
  --encoder-retention 0.5 \
  --encoder-prune-layer 0
```

### 5. UniTime adaptive with SemVID post-encoder pruning

```bash
hybrid-vtg run \
  --benchmark omtg \
  --data ./assets/datasets/omtg \
  --model unitime \
  --method unitime-adaptive \
  --corridor-top-k 4 \
  --subset 10 \
  --seed 42 \
  --post-pruning semvid \
  --post-retention 0.125
```

For the dense fixed control, use `--method unitime-fixed` without pruning
arguments. The Qwen2 backends compute only the final-token logits during
generation and cap evidence at 4,096 cells. A 24 GB GPU is still close to the
practical limit for a BF16 7B model; SemVID at 12.5% is the lowest-memory
configuration above.

The adapter-free Qwen models can run the same adaptive routing with Mage or
SemVID. These are ablations, not UniTime reproductions: the UniTime LoRA is
compatible with Qwen2-VL-7B only and cannot be loaded into Qwen3-VL-4B.
Qwen3 now uses the same explicit 4,096-cell evidence budget.

OMTG generation allows up to 256 new tokens so the model can enumerate many
separate occurrences; other benchmarks retain the 32-token limit. Predictions
are never joined merely because their timestamps are close. Postprocessing only
clips invalid boundaries and suppresses near-identical spans above 0.8 IoU.

```bash
# Base Qwen2-VL-7B, 4,096-cell budget, adaptive routing, and Mage
hybrid-vtg run --benchmark omtg --model qwen2-vl-7b \
  --method unitime-adaptive --corridor-top-k 8 --subset 10 --seed 42 \
  --encoder-pruning mage --encoder-retention 0.5 --encoder-prune-layer 0

# Base Qwen3-VL-4B, 4,096-cell budget, adaptive routing, and SemVID
hybrid-vtg run --benchmark omtg --model qwen3-vl-4b \
  --method unitime-adaptive --corridor-top-k 8 --subset 10 --seed 42 \
  --post-pruning semvid --post-retention 0.125
```

The first run downloads `Qwen/Qwen2-VL-7B-Instruct` and `zeqianli/UniTime` into
the Hugging Face cache. To use local weights, pass the base and adapter
separately:

```bash
hybrid-vtg run \
  --benchmark omtg \
  --model unitime \
  --method unitime-adaptive \
  --base-checkpoint /path/to/Qwen2-VL-7B-Instruct \
  --checkpoint /path/to/UniTime-adapter \
  --corridor-top-k 4 \
  --subset 10 \
  --seed 42
```

Runs are resumable and stored under
`results/runs/omtg/<model-variant>/<method-variant>/seed-42/`. Pruning settings
are part of `<model-variant>`, and adaptive top-k is part of `<method-variant>`,
so these five experiments cannot overwrite one another.

## Download datasets and checkpoints

One command downloads the three benchmarks, UniTime adapter plus its Qwen2-VL
base, TimeLens2-4B, TimeLens-7B, and the official UniVTG CLIP-B/32 checkpoint:

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
OMTG, TACoS, QVHighlights, UniTime, TimeLens, and TimeLens2 downloads. The token is never stored
in `assets/manifest.json`.

The equivalent repository script is:

```bash
python scripts/download_assets.py --root ./assets --accept-licenses
```

To download only selected assets, name them explicitly:

```bash
hybrid-vtg download tacos unitime timelens2-4b timelens-7b --root ./assets --accept-licenses
```

Valid targets are `omtg`, `tacos`, `qvhighlights`, `unitime`,
`timelens2-4b`, `timelens-7b`, and `univtg`. With no targets, all seven are
downloaded. The layout is:

```text
assets/
├── manifest.json
├── datasets/
│   ├── omtg/             OMTGBench.tsv + videos/
│   ├── tacos/            test annotations + 25 compressed test videos
│   └── qvhighlights/     test annotation + its 1,529 test videos only
└── checkpoints/
    ├── unitime/
    │   ├── adapter/
    │   └── qwen2-vl-7b/
    ├── timelens2-4b/
    ├── timelens-7b/
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
[TimeLens2](https://huggingface.co/MCG-NJU/TimeLens2-4B),
[TimeLens-7B](https://huggingface.co/TencentARC/TimeLens-7B),
[UniTime](https://huggingface.co/zeqianli/UniTime), and
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
| `qwen2-vl-7b` | `Qwen/Qwen2-VL-7B-Instruct` | Adapter-free Qwen2 baseline; 4,096-cell budget with adaptive routing, Mage, and SemVID |
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | 4,096-cell evidence budget with optional Mage and SemVID |
| `timelens2-4b` | `MCG-NJU/TimeLens2-4B` | Native 2 FPS or adaptive path, both capped at 4,096 visual tokens |
| `timelens-7b` | `TencentARC/TimeLens-7B` | Native timestamp or adaptive path, both capped at 4,096 visual tokens |
| `unitime` | `zeqianli/UniTime` adapter on `Qwen/Qwen2-VL-7B-Instruct` | Frozen reference/adaptive backend with optional Mage and SemVID policies |
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

The trimmed UniVTG transformer and positional-encoding implementation is derived from the official [showlab/UniVTG](https://github.com/showlab/UniVTG) repository under MIT. The fixed-budget method is a clean reimplementation of the TimeLens2 evaluation policy. Model checkpoints and datasets are not redistributed here and retain their own terms. Read [NOTICE.md](NOTICE.md) and [LICENSES](LICENSES/) before use. TimeLens2-4B declares Apache-2.0. The TimeLens-7B model card labels the checkpoint BSD-3-Clause, while its source repository includes additional TimeLens terms; review both upstream distributions before use.
