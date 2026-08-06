#!/usr/bin/env bash
set -euo pipefail

readonly OFFICIAL_PAGE="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus"
readonly DEFAULT_VIDEOS_URL="https://datasets.d2.mpi-inf.mpg.de/tacos/videos.zip"
readonly ANNOTATION_PAGE="https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos"
readonly DEFAULT_ANNOTATION_BASE="https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/resolve/main/tacos"
readonly VIDEO_ARCHIVE_BYTES=11281674033
readonly EXPECTED_VIDEOS=127

readonly -a SPLITS=(train val test)
readonly -a EXPECTED_QUERIES=(9790 4436 4001)
readonly -a ANNOTATION_SHA256=(
  d5c931511093d0e4a4a7f5b6592420ac1c33e2a824eb3024408f538593959e7f
  1f75348f28e3dcf1e4941eb3477cf7c7b6d587ca21c060d38a35b25029090107
  07e29508bfc2d512f1ec2f7b70b289ab094dd8ed8f39403fc42be14be62bb675
)

usage() {
  cat <<'EOF'
Prepare the full-fidelity TACoS temporal-grounding benchmark on a Linux server.

Usage:
  bash hybrid_vtg/scripts/prepare_tacos.sh --accept-license [options]

Options:
  --data-root DIR   Destination root (default: $TACOS_ROOT or ./data/TACoS)
  --keep-archive    Keep the downloaded 10.5 GiB ZIP after extraction
  --accept-license  Confirm that you accept the TACoS/MPII data terms (required)
  -h, --help        Show this help

Environment overrides for authorized mirrors:
  TACOS_VIDEOS_URL
  TACOS_ANNOTATION_BASE_URL

The script downloads all 127 original high-frame-rate AVI videos plus the
standard train/val/test VTG annotations. Allow at least 25 GiB of free disk
space during preparation. The archive is removed after successful validation.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command '$1'"
}

download() {
  local url="$1"
  local destination="$2"
  local label="$3"
  local partial="${destination}.incomplete"

  if [[ -s "$destination" ]]; then
    echo "[skip] ${label} already downloaded: ${destination}"
    return
  fi
  echo "[download] ${label}"
  echo "           ${url}"
  curl --fail --location \
    --retry 12 --retry-delay 10 --retry-all-errors \
    --continue-at - --output "$partial" "$url"
  mv "$partial" "$destination"
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die \
    "${label} checksum mismatch: expected ${expected}, got ${actual}. Remove ${path} and retry."
  echo "[OK] ${label} checksum"
}

data_root="${TACOS_ROOT:-}"
keep_archive=false
accepted=false

while (($#)); do
  case "$1" in
    --data-root)
      (($# >= 2)) || die "--data-root requires a directory"
      data_root="$2"
      shift 2
      ;;
    --keep-archive)
      keep_archive=true
      shift
      ;;
    --accept-license)
      accepted=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$accepted" == true ]] || {
  echo "Review the TACoS/MPII terms and sources first:"
  echo "  ${OFFICIAL_PAGE}"
  echo "  ${ANNOTATION_PAGE}"
  echo
  usage
  exit 2
}

for command_name in curl unzip sha256sum stat awk jq find sort comm ffprobe wc df mv rm date mkdir; do
  require_command "$command_name"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${data_root:-${repo_root}/data/TACoS}"
annotation_dir="${data_root}/annotations"
video_dir="${data_root}/videos"
download_dir="${data_root}/.downloads"
manifest_dir="${data_root}/manifests"
archive_path="${download_dir}/videos.zip"
videos_url="${TACOS_VIDEOS_URL:-$DEFAULT_VIDEOS_URL}"
annotation_base="${TACOS_ANNOTATION_BASE_URL:-$DEFAULT_ANNOTATION_BASE}"

mkdir -p "$annotation_dir" "$video_dir" "$download_dir" "$manifest_dir"

available_kib="$(df -Pk "$data_root" | awk 'NR==2 {print $4}')"
echo "[disk] $((available_kib / 1024 / 1024)) GiB currently free"
if ((available_kib < 25 * 1024 * 1024)); then
  echo "[warning] less than 25 GiB is free; preparation may run out of space" >&2
fi

expected_ids="${manifest_dir}/expected_video_ids.txt"
available_ids="${manifest_dir}/available_video_ids.txt"
missing_ids="${manifest_dir}/missing_video_ids.txt"
: > "$expected_ids"

total_queries=0
for index in "${!SPLITS[@]}"; do
  split="${SPLITS[$index]}"
  annotation_path="${annotation_dir}/${split}.jsonl"
  download "${annotation_base}/${split}.jsonl" "$annotation_path" "TACoS ${split} annotations"
  if [[ "$annotation_base" == "$DEFAULT_ANNOTATION_BASE" ]]; then
    verify_sha256 "$annotation_path" "${ANNOTATION_SHA256[$index]}" "${split} annotations"
  else
    echo "[warning] checksum skipped for ${split} annotations from custom mirror"
  fi

  expected_queries="${EXPECTED_QUERIES[$index]}"
  jq -e -s --argjson expected "$expected_queries" '
    length == $expected and all(.[];
      (.qid | type) == "string" and (.vid | type) == "string" and
      (.query | type) == "string" and (.duration | type) == "number" and
      (.relevant_windows | type) == "array" and
      (.relevant_windows | length) > 0 and
      all(.relevant_windows[]; type == "array" and length == 2))
  ' "$annotation_path" >/dev/null || die "invalid ${split} annotation schema or row count"
  jq -r '.vid' "$annotation_path" >> "$expected_ids"
  total_queries=$((total_queries + expected_queries))
  echo "[OK] ${split}: ${expected_queries} grounding queries"
done
sort -u -o "$expected_ids" "$expected_ids"

expected_video_count="$(wc -l < "$expected_ids")"
((expected_video_count == EXPECTED_VIDEOS)) || die \
  "expected ${EXPECTED_VIDEOS} unique annotated videos, found ${expected_video_count}"

find "$video_dir" -type f \( -iname '*.avi' -o -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' \) \
  -printf '%f\n' | awk '{sub(/[.][^.]+$/, ""); print}' | sort -u > "$available_ids"
existing_count="$(comm -12 "$expected_ids" "$available_ids" | wc -l)"

if ((existing_count == EXPECTED_VIDEOS)); then
  echo "[skip] all ${EXPECTED_VIDEOS} TACoS videos already exist"
else
  download "$videos_url" "$archive_path" "127 original TACoS videos (10.5 GiB)"
  if [[ "$videos_url" == "$DEFAULT_VIDEOS_URL" ]]; then
    actual_bytes="$(stat -c '%s' "$archive_path")"
    ((actual_bytes == VIDEO_ARCHIVE_BYTES)) || die \
      "video archive size mismatch: expected ${VIDEO_ARCHIVE_BYTES} bytes, found ${actual_bytes}. Remove ${archive_path} and retry."
    echo "[OK] video archive byte size"
  else
    echo "[warning] byte-size check skipped for custom video mirror"
  fi

  archive_manifest="${manifest_dir}/archive_members.txt"
  unzip -Z1 "$archive_path" > "$archive_manifest"
  if awk '
    /(^|[/\\])[.][.]([/\\]|$)/ || /^[/\\]/ || /^[A-Za-z]:[/\\]/ {bad=1}
    END {exit bad ? 0 : 1}
  ' "$archive_manifest"; then
    die "unsafe path found in TACoS ZIP; refusing to extract"
  fi
  unzip -tq "$archive_path" >/dev/null
  echo "[extract] original videos into ${video_dir}"
  unzip -nq "$archive_path" -d "$video_dir"
fi

find "$video_dir" -type f \( -iname '*.avi' -o -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' \) \
  -printf '%f\n' | awk '{sub(/[.][^.]+$/, ""); print}' | sort -u > "$available_ids"
comm -23 "$expected_ids" "$available_ids" > "$missing_ids"
available_count="$(comm -12 "$expected_ids" "$available_ids" | wc -l)"
missing_count="$(wc -l < "$missing_ids")"

if ((missing_count > 0)); then
  echo "Missing annotated TACoS video IDs (first 20):" >&2
  sed -n '1,20p' "$missing_ids" >&2
  die "dataset is incomplete: ${available_count}/${EXPECTED_VIDEOS} annotated videos found"
fi

first_video="$(find "$video_dir" -type f \( -iname '*.avi' -o -iname '*.mp4' \) -print -quit)"
[[ -n "$first_video" ]] || die "no TACoS video found after extraction"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 "$first_video" >/dev/null
echo "[OK] ${available_count} videos; sample video passes ffprobe"

jq -n \
  --arg source "$OFFICIAL_PAGE" \
  --arg annotations "$ANNOTATION_PAGE" \
  --arg prepared_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson videos "$available_count" \
  --argjson queries "$total_queries" \
  '{source:$source, annotations:$annotations, prepared_utc:$prepared_utc,
    videos:$videos, queries:$queries, default_split:"test", full_fidelity:true}' \
  > "${manifest_dir}/availability.json"

if [[ "$keep_archive" == false && -f "$archive_path" ]]; then
  rm -f "$archive_path"
  echo "[cleanup] removed the 10.5 GiB ZIP; it can be downloaded again"
fi

cat > "${data_root}/activate_tacos.sh" <<'EOF'
TACOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TACOS_ROOT
export TACOS_SPLIT=test
EOF

cat > "${data_root}/PREPARED.txt" <<EOF
TACoS temporal-grounding data prepared by Hybrid-VTG
source=${OFFICIAL_PAGE}
annotations=${ANNOTATION_PAGE}
videos=${available_count}
queries=${total_queries}
test_queries=${EXPECTED_QUERIES[2]}
prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo
echo "TACoS preparation complete."
echo "Dataset root: ${data_root}"
echo
echo "Next commands:"
printf '  source %q\n' "${data_root}/activate_tacos.sh"
echo "  bash hybrid_vtg/scripts/run_tacos_local.sh --limit 10 --fail-fast"
