#!/usr/bin/env bash
set -euo pipefail

readonly DATASET_REPO="insomnia7/omtg_bench"
readonly DATASET_REVISION="85b02b587a983bb1402890f49ff3ff92a1c02e9f"
readonly TSV_SHA256="938a5ae2e2486102f6dcf76260cc7ff7fe8f50388330862dcc40fd14d1cd035d"
readonly EXPECTED_ROWS=320
readonly EXPECTED_VIDEOS=287

usage() {
  cat <<'EOF'
Prepare the fixed OMTG Bench evaluation set.

Usage:
  bash hybrid_vtg/scripts/prepare_omtg.sh [options]

Options:
  --data-root DIR  Destination (default: $OMTG_ROOT or ./data/OMTGBench)
  --keep-archive   Keep the downloaded videos.zip after extraction
  -h, --help       Show this help
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${OMTG_ROOT:-${repo_root}/data/OMTGBench}"
keep_archive=false
while (($#)); do
  case "$1" in
    --data-root) (($# >= 2)) || die "--data-root requires a directory"; data_root="$2"; shift 2 ;;
    --keep-archive) keep_archive=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
for command_name in hf unzip awk find ffprobe mkdir mv cp wc sha256sum sort comm rm sed; do
  command -v "$command_name" >/dev/null 2>&1 || die "missing command: $command_name"
done

download_dir="${data_root}/.downloads"
video_dir="${data_root}/videos"
archive_path="${download_dir}/videos.zip"
expected_videos="${data_root}/expected-videos.txt"
available_videos="${data_root}/available-videos.txt"
mkdir -p "$download_dir" "$video_dir"

hf download "$DATASET_REPO" OMTGBench.tsv \
  --repo-type dataset --revision "$DATASET_REVISION" --local-dir "$download_dir"
cp "${download_dir}/OMTGBench.tsv" "${data_root}/OMTGBench.tsv.incomplete"
mv "${data_root}/OMTGBench.tsv.incomplete" "${data_root}/OMTGBench.tsv"
[[ "$(sha256sum "${data_root}/OMTGBench.tsv" | awk '{print $1}')" == "$TSV_SHA256" ]] || \
  die "OMTGBench.tsv checksum mismatch"
[[ "$(awk 'END {print NR-1}' "${data_root}/OMTGBench.tsv")" -eq "$EXPECTED_ROWS" ]] || \
  die "OMTGBench.tsv must contain 320 data rows"
awk -F '\t' 'NR > 1 {print $2}' "${data_root}/OMTGBench.tsv" | sort -u > "$expected_videos"
[[ "$(wc -l < "$expected_videos")" -eq "$EXPECTED_VIDEOS" ]] || \
  die "OMTGBench.tsv must reference 287 unique videos"

find "$video_dir" -type f -iname '*.mp4' -printf '%f\n' | sort -u > "$available_videos"
if [[ -n "$(comm -23 "$expected_videos" "$available_videos")" ]]; then
  hf download "$DATASET_REPO" videos.zip \
    --repo-type dataset --revision "$DATASET_REVISION" --local-dir "$download_dir"
  unzip -Z1 "$archive_path" | awk '
    /(^|[/\\])[.][.]([/\\]|$)/ || /^[/\\]/ || /^[A-Za-z]:[/\\]/ {bad=1}
    END {exit bad ? 1 : 0}' || die "unsafe path in videos.zip"
  unzip -tq "$archive_path" >/dev/null
  unzip -nq "$archive_path" -d "$data_root"
fi

find "$video_dir" -type f -iname '*.mp4' -printf '%f\n' | sort -u > "$available_videos"
missing="$(comm -23 "$expected_videos" "$available_videos")"
[[ -z "$missing" ]] || die "missing referenced OMTG videos; first: $(printf '%s\n' "$missing" | sed -n '1p')"
first_video="$(find "$video_dir" -type f -iname '*.mp4' -print -quit)"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$first_video" >/dev/null
printf '%s\n' "$DATASET_REVISION" > "${data_root}/dataset-revision.txt"
if [[ "$keep_archive" == false && -f "$archive_path" ]]; then
  rm -f "$archive_path"
fi
echo "[ready] ${data_root}: ${EXPECTED_ROWS} queries, ${EXPECTED_VIDEOS} videos"
echo "Set OMTG_ROOT=${data_root}"
