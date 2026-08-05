#!/usr/bin/env bash
set -euo pipefail

readonly DATASET_PAGE="https://huggingface.co/datasets/jwnt4/charades-sta-test"
readonly OFFICIAL_PAGE="https://prior.allenai.org/projects/charades"
readonly DEFAULT_ANNOTATION_URL="${DATASET_PAGE}/resolve/main/charades_sta_test.txt"
readonly DEFAULT_VIDEOS_URL="${DATASET_PAGE}/resolve/main/videos.zip"
readonly ANNOTATION_SHA256="fab341679c20b0917173dcffffde367a71113fe63f3a645558cef2ebfbfefb55"
readonly VIDEOS_SHA256="86857f45f73792d5d313680ddbe95608d510b3dc2316cddb8eeca95adbe8e950"
readonly EXPECTED_QUERIES=3720
readonly EXPECTED_VIDEOS=1334

usage() {
  cat <<'EOF'
Prepare the Charades-STA test benchmark on a headless server.

Usage:
  bash hybrid_vtg/scripts/prepare_charades_sta.sh --accept-license [options]

Options:
  --data-root DIR   Destination root (default: $CHARADES_STA_ROOT or ./data/Charades)
  --keep-archive    Keep the downloaded 6.2 GiB ZIP after successful extraction
  --accept-license  Confirm that you accept the Charades data license (required)
  -h, --help        Show this help

Environment overrides for mirrors:
  CHARADES_ANNOTATION_URL
  CHARADES_VIDEOS_URL

The default public test mirror contains 1,334 videos and 3,720 queries.
Approximately 15 GiB of free disk space is recommended during preparation.
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
  local partial="${destination}.part"
  local label="$3"

  if [[ -s "$destination" ]]; then
    echo "[skip] ${label} already downloaded: ${destination}"
    return
  fi
  echo "[download] ${label}"
  echo "           ${url}"
  curl --fail --location \
    --retry 8 --retry-delay 5 --retry-all-errors \
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

video_count() {
  find "$1" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.avi' \) \
    -printf '.' 2>/dev/null | wc -c
}

data_root="${CHARADES_STA_ROOT:-}"
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
  echo "Review the Charades license and terms first:"
  echo "  ${OFFICIAL_PAGE}"
  echo "  ${DATASET_PAGE}"
  echo
  usage
  exit 2
}

for command_name in curl unzip sha256sum awk find sort comm ffprobe sed wc df mktemp mv rm date; do
  require_command "$command_name"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${data_root:-${repo_root}/data/Charades}"
annotation_dir="${data_root}/sta_annotation"
video_dir="${data_root}/rgb_videos_30fps_480"
download_dir="${data_root}/.downloads"
annotation_path="${annotation_dir}/charades_sta_test.txt"
archive_path="${download_dir}/videos.zip"
annotation_url="${CHARADES_ANNOTATION_URL:-$DEFAULT_ANNOTATION_URL}"
videos_url="${CHARADES_VIDEOS_URL:-$DEFAULT_VIDEOS_URL}"

mkdir -p "$annotation_dir" "$video_dir" "$download_dir"

existing_videos="$(video_count "$video_dir")"
if ((existing_videos < EXPECTED_VIDEOS)); then
  available_kib="$(df -Pk "$data_root" | awk 'NR==2 {print $4}')"
  required_kib=15000000
  if ((available_kib < required_kib)); then
    die "insufficient disk space: need about 15 GiB free, found $((available_kib / 1024 / 1024)) GiB"
  fi
fi

download "$annotation_url" "$annotation_path" "Charades-STA annotations"
if [[ "$annotation_url" == "$DEFAULT_ANNOTATION_URL" ]]; then
  verify_sha256 "$annotation_path" "$ANNOTATION_SHA256" "annotation"
else
  echo "[warning] annotation checksum skipped for custom URL"
fi

query_count="$(awk 'NF {count++} END {print count+0}' "$annotation_path")"
invalid_queries="$(awk '
  NF && ($1 == "" || $2 !~ /^-?[0-9]+([.][0-9]+)?$/ || index($0, "##") == 0) {count++}
  END {print count+0}
' "$annotation_path")"
((invalid_queries == 0)) || die "annotation file contains ${invalid_queries} malformed rows"
((query_count == EXPECTED_QUERIES)) || die \
  "expected ${EXPECTED_QUERIES} annotation rows, found ${query_count}"
echo "[OK] ${query_count} temporal-grounding queries"

if ((existing_videos < EXPECTED_VIDEOS)); then
  download "$videos_url" "$archive_path" "Charades-STA test videos (about 6.2 GiB)"
  if [[ "$videos_url" == "$DEFAULT_VIDEOS_URL" ]]; then
    verify_sha256 "$archive_path" "$VIDEOS_SHA256" "video archive"
  else
    echo "[warning] video checksum skipped for custom URL"
  fi

  archive_manifest="$(mktemp)"
  missing_ids="$(mktemp)"
  annotation_ids="$(mktemp)"
  video_ids="$(mktemp)"
  cleanup() {
    rm -f "$archive_manifest" "$missing_ids" "$annotation_ids" "$video_ids"
  }
  trap cleanup EXIT

  unzip -Z1 "$archive_path" > "$archive_manifest"
  if awk '
    /(^|[/\\])[.][.]([/\\]|$)/ || /^[/\\]/ || /^[A-Za-z]:[/\\]/ {bad=1}
    END {exit bad ? 0 : 1}
  ' "$archive_manifest"; then
    die "unsafe path found in video ZIP; refusing to extract"
  fi
  unzip -tq "$archive_path" >/dev/null
  echo "[extract] videos into ${video_dir}"
  unzip -nq "$archive_path" -d "$video_dir"
else
  echo "[skip] found ${existing_videos} videos; archive download and extraction not needed"
  archive_manifest="$(mktemp)"
  missing_ids="$(mktemp)"
  annotation_ids="$(mktemp)"
  video_ids="$(mktemp)"
  cleanup() {
    rm -f "$archive_manifest" "$missing_ids" "$annotation_ids" "$video_ids"
  }
  trap cleanup EXIT
fi

find "$video_dir" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.avi' \) \
  -printf '%f\n' | sed -E 's/[.][^.]+$//' | sort -u > "$video_ids"
awk 'NF {print $1}' "$annotation_path" | sed -E 's/[.][^.]+$//' | sort -u > "$annotation_ids"
comm -23 "$annotation_ids" "$video_ids" > "$missing_ids"

final_video_count="$(wc -l < "$video_ids")"
if [[ -s "$missing_ids" ]]; then
  echo "Missing annotated video IDs (first 20):" >&2
  sed -n '1,20p' "$missing_ids" >&2
  die "dataset is incomplete; ${final_video_count} unique videos were found"
fi
((final_video_count >= EXPECTED_VIDEOS)) || die \
  "expected at least ${EXPECTED_VIDEOS} unique videos, found ${final_video_count}"

first_video="$(find "$video_dir" -type f -iname '*.mp4' -print -quit)"
[[ -n "$first_video" ]] || die "no MP4 video found after extraction"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height \
  -of default=noprint_wrappers=1 "$first_video" >/dev/null
echo "[OK] ${final_video_count} videos; sample video passes ffprobe"

if [[ "$keep_archive" == false && -f "$archive_path" ]]; then
  rm -f "$archive_path"
  echo "[cleanup] removed downloaded ZIP to recover disk space; it can be downloaded again"
fi

printf 'export CHARADES_STA_ROOT=%q\n' "$data_root" > "${data_root}/activate_charades.sh"
cat > "${data_root}/PREPARED.txt" <<EOF
Charades-STA test data prepared by Hybrid-VTG
source=${DATASET_PAGE}
queries=${query_count}
videos=${final_video_count}
annotation=${annotation_path}
prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo
echo "Charades-STA preparation complete."
echo "Dataset root: ${data_root}"
echo
echo "Next commands:"
printf '  source %q\n' "${data_root}/activate_charades.sh"
echo "  bash hybrid_vtg/scripts/run_charades_local.sh --limit 10 --fail-fast"
