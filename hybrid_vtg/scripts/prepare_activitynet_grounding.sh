#!/usr/bin/env bash
set -euo pipefail

readonly OFFICIAL_PAGE="https://activity-net.org/download.html"
readonly MIRROR_PAGE="https://huggingface.co/datasets/friedrichor/ActivityNet_Captions"
readonly DEFAULT_MIRROR_BASE="${MIRROR_PAGE}/resolve/main"
readonly DEFAULT_ANNOTATION_URL="${DEFAULT_MIRROR_BASE}/raw_data/val_2.json"
readonly ANNOTATION_SHA256="08fe56568bc92dd02a6150fe1094457a5f2ac72975b8778a922bc9a65aa7f0e9"
readonly EXPECTED_VIDEOS=4885
readonly EXPECTED_QUERIES=17031
readonly ARCHIVE_BYTES=41982156800

readonly -a ARCHIVE_PARTS=(
  ActivityNet_Videos.tar.part-000
  ActivityNet_Videos.tar.part-001
  ActivityNet_Videos.tar.part-002
  ActivityNet_Videos.tar.part-003
  ActivityNet_Videos.tar.part-004
  ActivityNet_Videos.tar.part-005
  ActivityNet_Videos.tar.part-006
  ActivityNet_Videos.tar.part-007
)

readonly -a ARCHIVE_SHA256=(
  cb6ef6065e148e4474a933b7b8fe99d32fd83bcb0ce7b5c0db6be814c414e6de
  898b739030dd358ff2cfa1efbc41becda6995a84db5d6ec43f979dd7eea7ea36
  61d6b89c8a8332ea7c79a49848eca7053a2822086e8b36f4d9152fa4c81c3c16
  625ad53d4c65e82fc1d8fc1d9d0c72b051d982b7da070d1fad3f5f616f6179b2
  6b2fd3e8161417ae6c38c2a3d95c39048c0ab65ee6b1318c6b413e11535300c3
  c7e05a67c455b7c55cd364e65e4b47e10faba23e0372b777105243712a926069
  337d10d13afe293f0a4d9c3181c5c77934c2619a31351926eec3a1d0b54c6896
  3b09f19c849cc90c23e83c919af12436a357eb6bea431b4fa19d5b978ab383fe
)

usage() {
  cat <<'EOF'
Prepare the ActivityNet-Grounding val_2 benchmark on a Linux GPU server.

Usage:
  bash hybrid_vtg/scripts/prepare_activitynet_grounding.sh --accept-license [options]

Options:
  --data-root DIR   Destination root (default: $ACTIVITYNET_ROOT or ./data/ActivityNet)
  --keep-archive    Keep the downloaded 39.1 GiB multipart TAR after extraction
  --accept-license  Confirm that you accept ActivityNet and source-video terms (required)
  -h, --help        Show this help

Environment overrides for an authorized mirror:
  ACTIVITYNET_MIRROR_BASE_URL
  ACTIVITYNET_ANNOTATION_URL

The public mirror is downloaded resumably and SHA-256 checked. The complete
archive is 39.1 GiB, but only the 4,885 val_2 videos are extracted. Allow at
least 60 GiB of free space during preparation. Archive parts are removed by
default after successful extraction, leaving a benchmark-ready dataset root.
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
  local partial="${destination}.incomplete"
  local label="$3"

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

human_gib() {
  awk -v bytes="$1" 'BEGIN {printf "%.1f", bytes / 1024 / 1024 / 1024}'
}

data_root="${ACTIVITYNET_ROOT:-}"
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
  echo "Review the ActivityNet and source-video terms first:"
  echo "  ${OFFICIAL_PAGE}"
  echo "  ${MIRROR_PAGE}"
  echo
  usage
  exit 2
}

for command_name in curl sha256sum awk jq tar find sort comm ffprobe wc df mv rm date mkdir; do
  require_command "$command_name"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${data_root:-${repo_root}/data/ActivityNet}"
caption_dir="${data_root}/captions"
video_dir="${data_root}/rgb_videos_15fps_short256"
download_dir="${data_root}/.downloads"
manifest_dir="${data_root}/manifests"
annotation_path="${caption_dir}/val_2.json"
available_annotation_path="${caption_dir}/val_2_available.json"
mirror_base="${ACTIVITYNET_MIRROR_BASE_URL:-$DEFAULT_MIRROR_BASE}"
annotation_url="${ACTIVITYNET_ANNOTATION_URL:-$DEFAULT_ANNOTATION_URL}"

mkdir -p "$caption_dir" "$video_dir" "$download_dir" "$manifest_dir"

available_kib="$(df -Pk "$data_root" | awk 'NR==2 {print $4}')"
available_bytes=$((available_kib * 1024))
echo "[disk] $(human_gib "$available_bytes") GiB currently free"
echo "[disk] multipart archive size: $(human_gib "$ARCHIVE_BYTES") GiB"
if ((available_bytes < 60000000000)); then
  echo "[warning] less than 60 GB is free; preparation may run out of space" >&2
fi

download "$annotation_url" "$annotation_path" "ActivityNet Captions val_2 annotations"
if [[ "$annotation_url" == "$DEFAULT_ANNOTATION_URL" ]]; then
  verify_sha256 "$annotation_path" "$ANNOTATION_SHA256" "annotation"
else
  echo "[warning] annotation checksum skipped for custom URL"
fi

jq -e 'type == "object" and all(.[]; (.duration | type) == "number" and (.timestamps | type) == "array" and (.sentences | type) == "array" and (.timestamps | length) == (.sentences | length))' \
  "$annotation_path" >/dev/null || die "annotation file has an unexpected schema"
video_count="$(jq 'keys | length' "$annotation_path")"
query_count="$(jq '[.[].sentences | length] | add' "$annotation_path")"
((video_count == EXPECTED_VIDEOS)) || die \
  "expected ${EXPECTED_VIDEOS} val_2 videos, found ${video_count}"
((query_count == EXPECTED_QUERIES)) || die \
  "expected ${EXPECTED_QUERIES} val_2 queries, found ${query_count}"
echo "[OK] ${video_count} annotated videos and ${query_count} grounding queries"

expected_ids="${manifest_dir}/expected_video_ids.txt"
available_ids="${manifest_dir}/available_video_ids.txt"
missing_ids="${manifest_dir}/missing_video_ids.txt"
expected_members="${manifest_dir}/expected_tar_members.txt"
archive_members="${manifest_dir}/archive_tar_members.txt"
selected_members="${manifest_dir}/selected_tar_members.txt"

jq -r 'keys[]' "$annotation_path" | sort -u > "$expected_ids"
jq -r 'keys[] | "Activity_Videos/\(.).mp4"' "$annotation_path" | sort -u > "$expected_members"
find "$video_dir" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
  | awk '{sub(/[.]mp4$/, ""); print}' | sort -u > "$available_ids"
existing_count="$(comm -12 "$expected_ids" "$available_ids" | wc -l)"

if ((existing_count == video_count)); then
  echo "[skip] all ${video_count} val_2 videos already exist; archive download is unnecessary"
else
  echo "[status] found ${existing_count}/${video_count} val_2 videos before extraction"
  for index in "${!ARCHIVE_PARTS[@]}"; do
    part="${ARCHIVE_PARTS[$index]}"
    part_path="${download_dir}/${part}"
    download "${mirror_base}/${part}" "$part_path" "archive part $((index + 1))/${#ARCHIVE_PARTS[@]}"
    if [[ "$mirror_base" == "$DEFAULT_MIRROR_BASE" ]]; then
      verify_sha256 "$part_path" "${ARCHIVE_SHA256[$index]}" "$part"
    else
      echo "[warning] checksum skipped for ${part} from custom mirror"
    fi
  done

  echo "[inspect] validating multipart TAR and indexing its members"
  tar -tf <(cat "${ARCHIVE_PARTS[@]/#/${download_dir}/}") | sort -u > "$archive_members"
  comm -12 "$expected_members" "$archive_members" > "$selected_members"

  selected_count="$(wc -l < "$selected_members")"
  ((selected_count > 0)) || die "none of the annotated val_2 videos exists in the archive"
  echo "[extract] ${selected_count}/${video_count} val_2 videos into ${video_dir}"
  tar -xf <(cat "${ARCHIVE_PARTS[@]/#/${download_dir}/}") \
    --directory "$video_dir" --strip-components=1 --files-from "$selected_members"
fi

find "$video_dir" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
  | awk '{sub(/[.]mp4$/, ""); print}' | sort -u > "$available_ids"
comm -23 "$expected_ids" "$available_ids" > "$missing_ids"

available_count="$(comm -12 "$expected_ids" "$available_ids" | wc -l)"
missing_count="$(wc -l < "$missing_ids")"
((available_count > 0)) || die "no annotated videos were extracted"

jq --rawfile ids "$available_ids" '
  ($ids | split("\n") | map(select(length > 0)) |
    reduce .[] as $id ({}; .[$id] = true)) as $available
  | with_entries(select($available[.key] == true))
' "$annotation_path" > "${available_annotation_path}.tmp"
mv "${available_annotation_path}.tmp" "$available_annotation_path"
available_queries="$(jq '[.[].sentences | length] | add // 0' "$available_annotation_path")"

first_video="$(find "$video_dir" -maxdepth 1 -type f -name '*.mp4' -print -quit)"
[[ -n "$first_video" ]] || die "no MP4 video found after extraction"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height \
  -of default=noprint_wrappers=1 "$first_video" >/dev/null
echo "[OK] sample video passes ffprobe"

jq -n \
  --arg source "$MIRROR_PAGE" \
  --arg split "val_2" \
  --arg prepared_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg annotation_sha256 "$(sha256sum "$annotation_path" | awk '{print $1}')" \
  --argjson expected_videos "$video_count" \
  --argjson available_videos "$available_count" \
  --argjson missing_videos "$missing_count" \
  --argjson expected_queries "$query_count" \
  --argjson available_queries "$available_queries" \
  '{source:$source, split:$split, prepared_utc:$prepared_utc,
    annotation_sha256:$annotation_sha256, expected_videos:$expected_videos,
    available_videos:$available_videos, missing_videos:$missing_videos,
    expected_queries:$expected_queries, available_queries:$available_queries}' \
  > "${manifest_dir}/availability.json"

if ((missing_count > 0)); then
  echo "[warning] ${missing_count} annotated videos are absent; see ${missing_ids}" >&2
else
  echo "[OK] all ${available_count} annotated val_2 videos are available"
fi

if [[ "$keep_archive" == false ]]; then
  for part in "${ARCHIVE_PARTS[@]}"; do
    rm -f "${download_dir}/${part}"
  done
  echo "[cleanup] removed the 39.1 GiB multipart archive; it can be downloaded again"
fi

cat > "${data_root}/activate_activitynet.sh" <<'EOF'
ACTIVITYNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACTIVITYNET_ROOT
export ACTIVITYNET_SPLIT=val_2_available
EOF
cat > "${data_root}/PREPARED.txt" <<EOF
ActivityNet-Grounding data prepared by Hybrid-VTG
source=${MIRROR_PAGE}
split=val_2
expected_videos=${video_count}
available_videos=${available_count}
missing_videos=${missing_count}
expected_queries=${query_count}
available_queries=${available_queries}
annotation=${annotation_path}
evaluation_annotation=${available_annotation_path}
prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo
echo "ActivityNet-Grounding preparation complete."
echo "Dataset root: ${data_root}"
echo
echo "Next commands:"
printf '  source %q\n' "${data_root}/activate_activitynet.sh"
echo "  bash hybrid_vtg/scripts/run_activitynet_local.sh --limit 10 --fail-fast"
