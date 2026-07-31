#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${TIMELENS2_DATA_ROOT:-$REPOSITORY_ROOT/data}"
BENCH_ROOT="${MOMENT_SEEKER_ROOT:-$DATA_ROOT/MomentSeeker}"
REVISION="${MOMENTSEEKER_REVISION:-a8fed5681192230df0f276aeaa6a4c14c8c685dd}"

if [[ "${MOMENTSEEKER_ACCEPT_LICENSE:-false}" != "true" ]]; then
  printf '%s\n' \
    'MomentSeeker is CC-BY-NC-SA-4.0 and restricted to research use.' \
    'Read https://huggingface.co/datasets/avery00/MomentSeeker and rerun with:' \
    '  MOMENTSEEKER_ACCEPT_LICENSE=true bash scripts/download_momentseeker.sh' >&2
  exit 2
fi
command -v hf >/dev/null 2>&1 || {
  printf 'error: Hugging Face CLI is required\n' >&2
  exit 2
}

mkdir -p "$BENCH_ROOT"
hf download avery00/MomentSeeker \
  --repo-type dataset \
  --revision "$REVISION" \
  --include 't2v.json' 'videos.tar.gz' 'videos.tar.gz.part_*' \
  --local-dir "$BENCH_ROOT"

BENCH_ROOT="$BENCH_ROOT" python - <<'PY'
import contextlib
import os
import tarfile
from pathlib import Path

root = Path(os.environ['BENCH_ROOT']).resolve()

def safe_target(name: str) -> Path:
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        raise SystemExit(f'unsafe archive path: {name}')
    return target

def extract_stream(file_object) -> None:
    with tarfile.open(fileobj=file_object, mode='r|gz') as bundle:
        for member in bundle:
            safe_target(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise SystemExit(f'unsupported archive entry: {member.name}')
            bundle.extract(member, root)

parts = sorted(root.glob('videos.tar.gz.part_*'))
archive = root / 'videos.tar.gz'
if parts:
    class ConcatenatedReader:
        def __init__(self, handles):
            self.handles = iter(handles)
            self.current = next(self.handles, None)

        def read(self, size=-1):
            chunks = []
            remaining = size
            while self.current is not None and (remaining < 0 or remaining > 0):
                chunk = self.current.read(-1 if remaining < 0 else remaining)
                if chunk:
                    chunks.append(chunk)
                    if remaining > 0:
                        remaining -= len(chunk)
                    continue
                self.current = next(self.handles, None)
            return b''.join(chunks)

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(part.open('rb')) for part in parts]
        extract_stream(ConcatenatedReader(handles))
elif archive.is_file():
    with archive.open('rb') as handle:
        extract_stream(handle)
else:
    raise SystemExit('MomentSeeker video archive or multipart archive is missing')

annotation = root / 't2v.json'
videos = root / 'videos'
if not annotation.is_file() or not videos.is_dir():
    raise SystemExit(
        'MomentSeeker layout is incomplete; expected t2v.json and videos/. '
        'Inspect the downloaded archive layout before running evaluation.'
    )
print(f'MomentSeeker ready at {root}')
PY

sha256sum "$BENCH_ROOT/t2v.json" > "$BENCH_ROOT/t2v.json.sha256"
printf '%s\n' "$REVISION" > "$BENCH_ROOT/dataset_revision.txt"
