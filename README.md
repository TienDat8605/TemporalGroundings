# TemporalGroundings GPU Server Runbook

This repository contains the training-free Timeline-Preserving Spatial Allocator (TPSA) and its full-video Qwen3-VL evaluation pipeline. This guide sets up a fresh rented NVIDIA GPU server, runs OMTG Bench first, and then runs the compressed VideoMind release of TACoS.

For allocator design and output details, see [`hybrid_vtg/README.md`](hybrid_vtg/README.md).

## Recommended server

- Ubuntu 22.04 or another recent x86-64 Linux distribution;
- one NVIDIA GPU with at least 24 GiB VRAM; 48 GiB gives more room for dense runs;
- a recent NVIDIA driver compatible with CUDA 12.x;
- Python 3.11;
- at least 60 GiB of persistent disk for the environment, model cache, OMTG, compressed TACoS, and outputs;
- stable internet access to GitHub and Hugging Face.

The PyTorch wheel includes its CUDA runtime, so a system CUDA toolkit is not required. The NVIDIA driver still must be working. PyTorch's official 2.8 packages provide CUDA 12.6, 12.8, and 12.9 builds: <https://pytorch.org/get-started/previous-versions/#v280>.

## 1. Connect and inspect the machine

SSH into the rental and start a persistent terminal before downloading or benchmarking:

```bash
ssh GPU_USER@GPU_HOST
tmux new -s tpsa
```

Inside `tmux`, verify the GPU and available storage:

```bash
nvidia-smi
df -h
```

Do not continue until `nvidia-smi` lists the rented GPU without an error. Detach from `tmux` with `Ctrl-b d`; reconnect later with `tmux attach -t tpsa`.

## 2. Install operating-system tools

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl ffmpeg git git-lfs jq tmux unzip
```

The dataset scripts require `ffprobe`, `jq`, `tar`, `unzip`, and standard GNU command-line tools.

## 3. Choose persistent paths

Replace `/workspace` with the rental's persistent volume. Avoid an ephemeral root filesystem when the provider offers attached storage.

```bash
export TPSA_WORKSPACE=/workspace
export TPSA_REPO="$TPSA_WORKSPACE/TemporalGroundings"
export TPSA_DATA="$TPSA_WORKSPACE/datasets"
export HF_HOME="$TPSA_WORKSPACE/huggingface-cache"
export OMTG_ROOT="$TPSA_DATA/OMTGBench"
export TACOS_ROOT="$TPSA_DATA/TACoS-compressed"

mkdir -p "$TPSA_WORKSPACE" "$TPSA_DATA" "$HF_HOME"
```

Re-export these variables after opening a new SSH or `tmux` session.

## 4. Clone the repository with submodules

```bash
git clone --recurse-submodules \
  https://github.com/TienDat8605/TemporalGroundings.git \
  "$TPSA_REPO"
cd "$TPSA_REPO"

git submodule update --init --recursive
git submodule status --recursive
```

Both `SemVID` and `TimeLens2/OMTG` should have a recorded commit next to their path. Do not omit `--recurse-submodules`: TPSA inherits SemVID's compact-prefill implementation.

## 5. Create the Python environment

Most GPU rental images already include Conda. If `conda` is unavailable, install Miniconda or use a provider image that includes Python 3.11.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n hybrid-vtg python=3.11 -y
conda activate hybrid-vtg

python -m pip install --upgrade pip setuptools wheel
```

Install the CUDA build of PyTorch first. CUDA 12.8 is the preferred match for the pinned PyTorch 2.8 stack:

```bash
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

If the provider explicitly supplies only a CUDA 12.6-compatible driver, use the official `cu126` index instead:

```bash
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Then install the repository dependencies and editable package:

```bash
python -m pip install -r hybrid_vtg/requirements.txt
python -m pip install -e 'hybrid_vtg[test]'
```

FlashAttention is optional. The launchers use PyTorch SDPA by default, so do not install FlashAttention just for the initial benchmark.

## 6. Authenticate with Hugging Face

The model and datasets are downloaded from Hugging Face. Log in with a read token to avoid anonymous rate limits:

```bash
hf auth login
hf auth whoami
```

Paste the token only into the interactive prompt; do not put it in a script or commit it. The official CLI documentation is at <https://huggingface.co/docs/huggingface_hub/en/guides/cli>.

For slow links, increase download timeouts:

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
```

## 7. Apply the local SemVID plumbing patches

```bash
cd "$TPSA_REPO"
bash hybrid_vtg/scripts/apply_semvid_patches.sh
```

The operation is idempotent: rerunning it detects patches that are already applied. It intentionally makes the `SemVID` submodule appear locally modified; benchmark code remains stored as patch files in the parent repository.

## 8. Validate the installation

Check the exact PyTorch build and CUDA visibility:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("wheel CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(index, torch.cuda.get_device_name(index), f"{props.total_memory / 2**30:.1f} GiB")
PY

hybrid-vtg doctor
PYTHONPATH=hybrid_vtg/src pytest -q hybrid_vtg/tests
```

Every `doctor` entry, including `decord` and CUDA, should report `OK`. The unit suite does not load Qwen weights.

## 9. Prepare OMTG Bench first

Review the [OMTG Bench dataset card](https://huggingface.co/datasets/insomnia7/omtg_bench), its CC BY-NC 4.0 terms, and the source-video licenses before downloading.

```bash
cd "$TPSA_REPO"
bash hybrid_vtg/scripts/prepare_omtg.sh --data-root "$OMTG_ROOT"
```

The downloader is pinned to a dataset revision, verifies the 320-row annotation checksum, checks archive paths before extraction, confirms all 287 referenced videos, and removes `videos.zip` after successful preparation unless `--keep-archive` is supplied.

Expected layout:

```text
$OMTG_ROOT/
├── OMTGBench.tsv
├── dataset-revision.txt
└── videos/
    └── *.mp4
```

## 10. Run an OMTG smoke test

The first invocation also downloads `Qwen/Qwen3-VL-4B-Thinking` into `$HF_HOME`.

```bash
mkdir -p "$TPSA_REPO/outputs/smoke"

bash hybrid_vtg/scripts/run_omtg_local.sh \
  --limit 4 \
  --spatial-policy tpsa_boundary \
  --retention-ratio 0.125 \
  --output "$TPSA_REPO/outputs/smoke/omtg-tpsa-boundary.jsonl" \
  --fail-fast
```

Confirm that the following files exist:

```bash
ls -lh "$TPSA_REPO/outputs/smoke/omtg-tpsa-boundary"*
```

The JSONL contains per-sample predictions and telemetry. The adjacent manifest freezes the configuration, and the metrics file reports OMTG C-Acc, tIoU, tF1, and EtF1.

## 11. Run the 12.5% OMTG comparison

The launcher is sequential and runs only `tpsa_query`, `tpsa_motion`, and `tpsa_boundary` at 12.5% retention. It deliberately skips dense, uniform, and SemVID because their baseline results are already available. At seven seconds per sample, the three 320-sample runs take approximately 1 hour 52 minutes, plus model-loading overhead.

```bash
mkdir -p "$TPSA_REPO/outputs/tpsa-matrix/omtg" "$TPSA_REPO/logs"

bash hybrid_vtg/scripts/run_spatial_matrix.sh \
  omtg "$TPSA_REPO/outputs/tpsa-matrix/omtg" \
  --fail-fast \
  2>&1 | tee "$TPSA_REPO/logs/omtg-matrix.log"
```

Run the command inside `tmux`. If SSH disconnects, the process continues. Result files are append-only and resume by sample ID, so rerunning the exact command with the same output directory continues incomplete jobs. Do not reuse an output file with different arguments because its manifest is immutable.

## 12. Prepare compressed TACoS second

This repository intentionally uses VideoMind's compressed TACoS artifact, `videos_3fps_480_noaudio.tar.gz`—3 fps, 480p, no audio, approximately 1.49 GB—rather than the 30.2 GB original-video archive. The source provides `train.jsonl`, `val.jsonl`, and `test.jsonl`: <https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos>.

```bash
bash hybrid_vtg/scripts/prepare_tacos.sh --data-root "$TACOS_ROOT"
```

The script validates annotation hashes and row counts, verifies all 127 videos, checks the frame-rate/resolution envelope, and rejects audio streams.

Run a smoke test and then the matrix:

```bash
bash hybrid_vtg/scripts/run_tacos_local.sh \
  --limit 4 \
  --spatial-policy tpsa_boundary \
  --retention-ratio 0.125 \
  --output "$TPSA_REPO/outputs/smoke/tacos-tpsa-boundary.jsonl" \
  --fail-fast

bash hybrid_vtg/scripts/run_spatial_matrix.sh \
  tacos "$TPSA_REPO/outputs/tpsa-matrix/tacos" \
  --fail-fast \
  2>&1 | tee "$TPSA_REPO/logs/tacos-matrix.log"
```

## Optional optimized execution

The safe profile uses Qwen batch size one. Only enable batch-two and CPU prefetch after checking equivalence on the rented GPU:

```bash
bash hybrid_vtg/scripts/run_omtg_local.sh \
  --limit 32 \
  --capture-validation-logits \
  --output "$TPSA_REPO/outputs/validation/batch-1.jsonl"

bash hybrid_vtg/scripts/run_omtg_local.sh \
  --limit 32 \
  --optimization-profile optimized \
  --capture-validation-logits \
  --output "$TPSA_REPO/outputs/validation/batch-2.jsonl"

hybrid-vtg validate-optimization \
  --baseline "$TPSA_REPO/outputs/validation/batch-1.jsonl" \
  --candidate "$TPSA_REPO/outputs/validation/batch-2.jsonl" \
  --minimum-samples 32 \
  --logit-tolerance 0.05
```

Use fresh output names for the optimized comparison. A batch-two CUDA out-of-memory fallback invalidates a strict speed comparison.

## Multi-GPU selection

The loader can use the visible GPUs. Restrict or select devices before launching:

```bash
export CUDA_VISIBLE_DEVICES=0
```

For two GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

Record the selected devices and GPU model with every benchmark report.

## Troubleshooting

### `hybrid-vtg doctor` reports CUDA unavailable

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

If the installed torch version ends in `+cpu` or `torch.version.cuda` is `None`, reinstall the official CUDA wheel from step 5. If `nvidia-smi` fails, repair or replace the rental image before changing Python packages.

### `decord` is missing

```bash
python -m pip install decord==0.6.0
hybrid-vtg doctor
```

### `ffmpeg` or `ffprobe` is missing

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### Hugging Face downloads time out

Re-export the timeout variables from step 6 and rerun the same preparation command. `hf download` resumes cached files rather than restarting completed objects.

### CUDA out of memory

1. Run `nvidia-smi` and stop unrelated GPU jobs.
2. Keep the default safe profile and Qwen batch size one.
3. Smoke-test `semvid` or `tpsa_boundary` before attempting the dense baseline.
4. If sampling settings must be reduced, restart every compared policy with the identical settings and new output paths.

### SemVID patch application fails

Inspect the submodule first:

```bash
git -C SemVID status --short
git submodule status SemVID
```

On a fresh clone, rerunning `bash hybrid_vtg/scripts/apply_semvid_patches.sh` is safe. If the submodule contains unrelated edits, preserve them or use a separate clean clone rather than overwriting them.

## Save results before terminating the rental

Outputs are ignored by Git and are not pushed with source commits. Copy them to durable storage before destroying the server. From your local machine:

```bash
scp -r \
  GPU_USER@GPU_HOST:/workspace/TemporalGroundings/outputs \
  ./server-outputs
```

Also preserve logs and the generated `*.manifest.json` and `*.metrics.json` files. These are required to verify that methods used identical frames, prompts, and retained-token budgets.

## Updating an existing server clone

```bash
cd "$TPSA_REPO"
git pull --ff-only
git submodule update --init --recursive
conda activate hybrid-vtg
python -m pip install -r hybrid_vtg/requirements.txt
python -m pip install -e 'hybrid_vtg[test]'
bash hybrid_vtg/scripts/apply_semvid_patches.sh
hybrid-vtg doctor
```

Keep old result directories unchanged and use new output paths when the project revision or configuration changes.
