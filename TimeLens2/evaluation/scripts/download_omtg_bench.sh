#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${TIMELENS2_DATA_ROOT:-$REPOSITORY_ROOT/data}"
OMTG_ROOT="${OMTG_BENCH_ROOT:-$DATA_ROOT/OMTGBench}"
MARKER="$OMTG_ROOT/.complete"
OMTG_REVISION="${OMTG_DATASET_REVISION:-85b02b587a983bb1402890f49ff3ff92a1c02e9f}"

validate_dataset() {
  OMTG_ROOT="$OMTG_ROOT" python - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ['OMTG_ROOT'])
tsv = root / 'OMTGBench.tsv'
videos = root / 'videos'
if not tsv.is_file() or not videos.is_dir():
    raise SystemExit(1)
with tsv.open(encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle, delimiter='\t'))
if len(rows) != 320:
    raise SystemExit(f'expected 320 OMTG rows, found {len(rows)}')
missing = sorted({row['video'] for row in rows if not (videos / row['video']).is_file()})
if missing:
    raise SystemExit(f'missing {len(missing)} referenced videos; first: {missing[0]}')
print(f'OMTG Bench ready: {len(rows)} queries, {len({row["video"] for row in rows})} videos at {root}')
PY
}

if validate_dataset 2>/dev/null; then
  sha256sum "$OMTG_ROOT/OMTGBench.tsv" > "$OMTG_ROOT/OMTGBench.tsv.sha256"
  if [[ ! -f "$OMTG_ROOT/dataset_revision.txt" ]]; then
    printf '%s\n' "unknown-existing" > "$OMTG_ROOT/dataset_revision.txt"
  fi
  touch "$MARKER"
  exit 0
fi

mkdir -p "$OMTG_ROOT"
command -v hf >/dev/null 2>&1 || {
  printf 'error: Hugging Face CLI is unavailable; run the evaluation setup first\n' >&2
  exit 2
}
hf download insomnia7/omtg_bench OMTGBench.tsv videos.zip \
  --repo-type dataset \
  --revision "$OMTG_REVISION" \
  --local-dir "$OMTG_ROOT"

OMTG_ROOT="$OMTG_ROOT" python - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ['OMTG_ROOT']).resolve()
archive = root / 'videos.zip'
if not archive.is_file():
    raise SystemExit(f'missing archive: {archive}')
with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        target = (root / member.filename).resolve()
        if target != root and root not in target.parents:
            raise SystemExit(f'unsafe path in videos.zip: {member.filename}')
    bundle.extractall(root)
PY

validate_dataset
sha256sum "$OMTG_ROOT/OMTGBench.tsv" > "$OMTG_ROOT/OMTGBench.tsv.sha256"
printf '%s\n' "$OMTG_REVISION" > "$OMTG_ROOT/dataset_revision.txt"
touch "$MARKER"
if [[ "${OMTG_KEEP_ARCHIVE:-0}" != "1" ]]; then
  rm -f "$OMTG_ROOT/videos.zip"
fi
