#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${TIMELENS2_DATA_ROOT:-$REPOSITORY_ROOT/data}"
BENCH_ROOT="${TIMELENS_BENCH_ROOT:-$DATA_ROOT/TimeLens-Bench}"
REVISION="${TIMELENS_BENCH_REVISION:-main}"

command -v hf >/dev/null 2>&1 || {
  printf 'error: Hugging Face CLI is required\n' >&2
  exit 2
}

mkdir -p "$BENCH_ROOT"
hf download TencentARC/TimeLens-Bench \
  --repo-type dataset \
  --revision "$REVISION" \
  --include 'qvhighlights-timelens.json' 'video_shards/qvhighlights/*.tar.gz' \
  --local-dir "$BENCH_ROOT"

BENCH_ROOT="$BENCH_ROOT" python - <<'PY'
import os
import tarfile
from pathlib import Path

root = Path(os.environ['BENCH_ROOT']).resolve()
destination = root / 'videos'
destination.mkdir(parents=True, exist_ok=True)
for archive in sorted((root / 'video_shards' / 'qvhighlights').glob('*.tar.gz')):
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise SystemExit(f'unsafe path in {archive}: {member.name}')
            if member.issym() or member.islnk() or member.isdev():
                raise SystemExit(f'unsupported archive entry in {archive}: {member.name}')
        bundle.extractall(destination)

annotation = root / 'qvhighlights-timelens.json'
videos = destination / 'qvhighlights'
if not annotation.is_file() or not videos.is_dir():
    raise SystemExit('TimeLens-Bench download is missing QVHighlights annotations or videos')
print(f'TimeLens QVHighlights ready at {root}')
PY

sha256sum "$BENCH_ROOT/qvhighlights-timelens.json" \
  > "$BENCH_ROOT/qvhighlights-timelens.json.sha256"
printf '%s\n' "$REVISION" > "$BENCH_ROOT/dataset_revision.txt"
