# 📊 TimeLens2 Evaluation

We employ [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate performance on the video temporal grounding benchmark.

## 🛠️ Install

```bash
pip install -e .
pip install -U flash-attn --no-build-isolation
```

## 🎞️ Benchmark roots

Set the roots for the benchmarks you want to run:

```bash
export TIMELENS_BENCH_ROOT=/path/to/TimeLens-Bench
export VUE_TR_ROOT=/path/to/VUE_TR
export VUE_TR_V2_ROOT=/path/to/VUE_TR_V2
export MOMENT_SEEKER_ROOT=/path/to/MomentSeeker
export EGO4D_NLQ_V2_ROOT=/path/to/Ego4D-NLQ-v2/annotations
export EGO4D_NLQ_V2_VIDEOS_DIR=/path/to/Ego4D/videos
export OMTG_BENCH_ROOT=/path/to/OMTGBench
```

The TimeLens benchmark root contains the Charades, ActivityNet, and
QVHighlights subsets expected by `vlmeval/dataset/timelens.py`.

OMTG Bench is the primary multi-span benchmark for the hierarchical-search
experiment. `TimeLens2-93K` is the training corpus that addresses supervision
quality; it is not used as an evaluation set. Download OMTG's 320 queries and
287 videos with:

```bash
TIMELENS2_DATA_ROOT=/path/to/data bash scripts/download_omtg_bench.sh
```

The downloader validates every TSV reference and writes a completion marker,
so rerunning it in the same runtime is inexpensive.

`LMUData` must point to a writable location. VLMEvalKit stores metadata, cached
frames, and intermediate files there, so a shared path is recommended when jobs
may be resumed or inspected from another machine.

## 🚀 Run

```bash
bash scripts/srun_eval_all/run_grounding.sh
```

The default command evaluates **both TimeLens2-4B and TimeLens2-8B on all seven datasets**.
For a local checkpoint, replace the corresponding `model_path` in
`vlmeval/config.py`, then set the model alias, dataset, and a
checkpoint-specific output directory explicitly:

```bash
MODELS="TimeLens2-4B" \
DATASETS="TimeLens_Charades_4fps" \
N_GPU=8 \
CHECK_EXTRACTED_FRAMES=False \
OUTPUT_DIR=outputs/timelens2-4b \
bash scripts/srun_eval_all/run_grounding.sh
```

Run one model or a subset of datasets with space-separated overrides:

```bash
MODELS="TimeLens2-4B" \
DATASETS="VUE_TR_V2_1fps_limit_2048_px480_ctx128k" \
bash scripts/srun_eval_all/run_grounding.sh
```

The entry point exposes `USE_LLM_JUDGE=auto|true|false`. Its default is `auto`:
the three TimeLens benchmarks use exact parsing, while VUE-TR, VUE-TR-V2,
MomentSeeker, and Ego4D-NLQ use the configured LLM judge.

```bash
USE_LLM_JUDGE=auto \
LLM_JUDGE="qwen3-235b-a22b-thinking-2507" \
DATASETS="TimeLens_Charades_4fps Ego4D-NLQ-v2_2fps_limit_2048_px480_ctx128k" \
bash scripts/srun_eval_all/run_grounding.sh
```

Set `USE_LLM_JUDGE=true` to force the judge for every selected benchmark, or
`USE_LLM_JUDGE=false` to disable it everywhere. The
example judge is the one used in our environment; its served-model alias
defaults to `LOCAL_LLM=qwen3-235b`. For another judge service, set `LLM_JUDGE`
and `LOCAL_LLM` as required by that service.

## Run from the Colab CLI

`scripts/colab_experiment.sh` packages only the current local code, uploads it
to a versioned directory on a T4 Colab VM, downloads the dataset on that VM,
starts the experiment as a background process, and monitors its log. It uses
`google-colab-cli` directly and does not mount Google Drive.
Before uploading a job, it probes the runtime file API and automatically
recreates a stale session that is still listed by the CLI but has lost its VM.

Put credentials and remote data locations in a local environment file that is
not included in the source archive. Put `.env.colab` in the TimeLens2 repository
root (beside `.gitignore`, not inside `evaluation/`), for example:

```bash
HF_TOKEN=your_token
VUE_TR_V2_ROOT=/content/timelens2-data/VUE_TR_V2
LMUData=/content/timelens2-data/LMUData
OMTG_BENCH_ROOT=/content/timelens2-data/OMTGBench
```

Launch a single-GPU run and follow its output:

```bash
evaluation/scripts/colab_experiment.sh start \
  --env-file .env.colab \
  --dataset-command 'bash scripts/download_vue_tr_v2.sh' \
  --command 'MODELS="TimeLens2-4B" DATASETS="VUE_TR_V2_1fps_limit_64_px336_ctx8k_t4" N_GPU=1 bash scripts/srun_eval_all/run_grounding.sh'
```

The VUE-TR-V2 downloader retrieves the official annotations from
`bytedance/vidi`, then downloads the listed YouTube videos at up to 480p with
`yt-dlp`. It performs a one-video access check before starting the batch so a
blocked Colab IP fails quickly. Browser sessions exported on a different
machine can be rotated or rejected from a Colab IP. If cookies are required,
pass a Mozilla/Netscape-format file with `--youtube-cookies`; the runner uploads
it outside the source archive with mode 0600 and deletes the remote copy when
the job finishes. The optional BgUtils PO-token provider can be enabled with
`YTDLP_ENABLE_POT_PROVIDER=true`, but it cannot guarantee bypassing an IP-level
bot block. Keep the local cookie file private—cookies are account credentials,
and a disposable YouTube account is safer. Set `VUE_TR_MAX_VIDEOS=20` in `.env.colab` for a small smoke-test
subset; use `0` or omit it for all listed videos. The command runs after setup
with `TIMELENS2_DATA_ROOT=/content/timelens2-data`, and existing video files are
reused when a session is resumed. Use `--no-dataset-download` only for tests
that require no benchmark data.

The default setup command installs the evaluation package, including the
reproducible Qwen3-VL stack (`transformers==4.57.6`,
`huggingface_hub==0.36.2`, `qwen_vl_utils==0.0.14`). TimeLens2 selects
PyTorch SDPA on T4 rather than requiring FlashAttention 2. Override setup with
`--setup-command`, or use `--no-setup` when reusing a prepared session. Job
control is available without restarting the experiment:

The Colab worker detects T4 GPUs and enables a memory-safe VUE-TR configuration:
64 uniformly sampled frames, a 336-pixel maximum edge, and an 8k visual-token
budget. If the full `VUE_TR[_V2]_1fps_limit_2048_px480_ctx128k` alias is passed,
the grounding script remaps it to the corresponding `*_64_px336_ctx8k_t4`
alias and logs the remap. The limits can be tightened in `.env.colab` with
`TIMELENS2_T4_MAX_FRAMES`, `TIMELENS2_T4_MAX_EDGE`, and
`TIMELENS2_T4_MAX_VIDEO_TOKENS`.

```bash
evaluation/scripts/colab_experiment.sh status
evaluation/scripts/colab_experiment.sh monitor
evaluation/scripts/colab_experiment.sh logs --lines 100
evaluation/scripts/colab_experiment.sh fetch
evaluation/scripts/colab_experiment.sh cancel  # cancel the job, keep the VM
evaluation/scripts/colab_experiment.sh stop    # release the Colab VM
```

`fetch` downloads the configured output paths, `evaluation/outputs` by default,
as a compressed archive. Use repeated `--output-path` options on `start` to
include other repository-relative result directories. Pressing Ctrl-C while
monitoring only detaches; it does not cancel the remote process.
At every terminal state, the monitor performs a final download and prints the
local `job.log` and `status.json` paths under `.colab-runs/<session>/<job-id>/`.
OMTG runs also create a compact checkpoint every 30 seconds and the monitor
downloads it to the same local job directory. If the remote job files disappear
for six consecutive checks, the monitor records `LOST` and exits instead of
polling forever. Colab storage is ephemeral, so fetch results before stopping
the session.

## OMTG hierarchical search experiment

`run_omtg_search.py` implements the frozen-model experiment in two GPU phases:

1. Generate deterministic 20--60 second content-aware windows (uniform
   fallback), encode four frames per window plus the query with
   `Qwen/Qwen3-VL-Embedding-2B`, and save the top
   `K=min(8,max(2,ceil(sqrt(N))))` routes.
2. Unload the embedding model, load `MCG-NJU/TimeLens2-4B`, run the four search
   schedules, map local spans to global time, suppress duplicates, and merge
   gaps of at most one second.

The schedules are `uniform-one-shot`, `full-video-multipass`,
`uniform-window-local`, and `embedding-window-local`. Router frames count
against the embedding schedule's nominal 32/64-frame budget. The result also
records model calls, synchronized single-GPU model time, wall time, peak VRAM,
budget overflow, router recall, and offline oracle router recall. A comparison
is marked compute-matched only when aggregate GPU time is within 5%; otherwise
the result is a Pareto comparison.

### Run locally

The local launcher supports any NVIDIA GPU with enough memory for the 2B router
and 4B grounder (they are loaded sequentially). It is single-GPU, but does not
require a T4 or require the machine to have exactly one GPU.

Install the evaluation environment and download OMTG Bench once:

```bash
cd evaluation
pip install -e .
cd ..
bash evaluation/scripts/download_omtg_bench.sh
```

Then run the deterministic 25-query smoke test:

```bash
bash evaluation/scripts/run_omtg_search_local.sh
```

By default, data is stored under `TimeLens2/data`, frame caches under
`evaluation/.cache`, and resumable results under
`evaluation/outputs/omtg_search/smoke`. Re-running the same command resumes
from the append-only route and prediction files.

Select a GPU or override the run configuration with environment variables:

```bash
CUDA_VISIBLE_DEVICES=1 \
OMTG_RUN_NAME=full \
OMTG_MAX_SAMPLES=0 \
OMTG_BUDGETS=32,64 \
bash evaluation/scripts/run_omtg_search_local.sh
```

Local model directories are supported:

```bash
OMTG_MODEL=/models/TimeLens2-4B \
OMTG_EMBEDDING_MODEL=/models/Qwen3-VL-Embedding-2B \
bash evaluation/scripts/run_omtg_search_local.sh
```

Use a new `OMTG_RUN_NAME` after changing models, budgets, schedules, sample
count, or paths. To retain decoded frame caches after an `all` run, append
`--keep-frame-cache`. Individual phases can be run with
`OMTG_PHASE=route`, `ground`, or `evaluate`.

### Run on Colab

Run the required deterministic 25-query smoke test on Colab:

```bash
evaluation/scripts/colab_experiment.sh start \
  --env-file .env.colab \
  --dataset-command 'bash scripts/download_omtg_bench.sh' \
  --command 'OMTG_RUN_NAME=smoke OMTG_MAX_SAMPLES=25 bash scripts/run_omtg_search.sh'
```

After the smoke run succeeds, run all 320 queries:

```bash
evaluation/scripts/colab_experiment.sh start \
  --env-file .env.colab \
  --dataset-command 'bash scripts/download_omtg_bench.sh' \
  --command 'OMTG_RUN_NAME=full OMTG_MAX_SAMPLES=0 bash scripts/run_omtg_search.sh'
```

The authoritative routes and predictions are append-only JSONL files under
`/content/timelens2-experiment-outputs/omtg_search/<run-name>/`, outside each
versioned job checkout but inside the active Colab VM. This makes a new `start`
resume the previous job without Google Drive. An exit trap mirrors that state to
`evaluation/outputs/omtg_search/<run-name>/` for the normal output fetch, even
when the experiment command fails. While the job is running, the wrapper also
writes `omtg_checkpoint.tar.gz` into its Colab job directory; the local monitor
downloads an atomic copy under `.colab-runs`.

If a VM is recycled, restart the identical experiment with the checkpoint path
printed by the monitor:

```bash
evaluation/scripts/colab_experiment.sh start \
  --env-file .env.colab \
  --resume-checkpoint .colab-runs/timelens2/JOB_ID/omtg_checkpoint.tar.gz \
  --dataset-command 'bash scripts/download_omtg_bench.sh' \
  --command 'OMTG_RUN_NAME=smoke OMTG_MAX_SAMPLES=25 bash scripts/run_omtg_search.sh'
```

The restored append-only files make the experiment continue only missing
records; keep the run name, sample count, budgets, and schedules identical.
For manual recovery, set `OMTG_PHASE=route`,
`ground`, or `evaluate`. At a terminal state the Colab wrapper always downloads
the final log/status to `.colab-runs`; run
`evaluation/scripts/colab_experiment.sh fetch` before stopping the VM to fetch
the mirrored experiment outputs.

`run.py` is called with `--reuse`, so do
not reuse the same `OUTPUT_DIR` and model alias for a different checkpoint unless
you intentionally want existing predictions to be reused.

`CHECK_EXTRACTED_FRAMES=True` validates cached frames before inference. Set it to `False` when decoding videos directly or when
the frame cache has already been validated. For repeated runs, pre-extracting
frames once is much faster:

```bash
python scripts/pre_extract_video_frames/extract_video_frames.py \
  --dataset TimeLens_Charades_4fps
```
