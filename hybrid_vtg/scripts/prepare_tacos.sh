#!/usr/bin/env bash
set -euo pipefail

readonly DATASET_REPO="yeliudev/VideoMind-Dataset"
readonly DATASET_REVISION="3518d8e8c2c7c1ceec4b242afc07162bbafa33d8"
readonly DATASET_PAGE="https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/tacos"
readonly ARCHIVE_NAME="videos_3fps_480_noaudio.tar.gz"
readonly EXPECTED_VIDEOS=127
readonly EXPECTED_ARCHIVE_VIDEOS=273
readonly -a SPLITS=(train val test)
readonly -a EXPECTED_QUERIES=(9790 4436 4001)
readonly -a ANNOTATION_SHA256=(
  d5c931511093d0e4a4a7f5b6592420ac1c33e2a824eb3024408f538593959e7f
  1f75348f28e3dcf1e4941eb3477cf7c7b6d587ca21c060d38a35b25029090107
  07e29508bfc2d512f1ec2f7b70b289ab094dd8ed8f39403fc42be14be62bb675
)

usage() {
  cat <<'EOF'
Prepare VideoMind's compressed TACoS benchmark (3 fps, 480p, no audio).

Usage:
  bash hybrid_vtg/scripts/prepare_tacos.sh [options]

Options:
  --data-root DIR  Destination (default: $TACOS_ROOT or ./data/TACoS-compressed)
  --keep-archive   Keep the 1.49 GB archive after verified extraction
  -h, --help       Show this help

Requires the Hugging Face `hf` CLI. The script downloads the three JSONL
splits and videos_3fps_480_noaudio.tar.gz from the pinned VideoMind revision.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command '$1'"; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${TACOS_ROOT:-${repo_root}/data/TACoS-compressed}"
keep_archive=false
while (($#)); do
  case "$1" in
    --data-root) (($# >= 2)) || die "--data-root requires a directory"; data_root="$2"; shift 2 ;;
    --keep-archive) keep_archive=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for command_name in hf tar jq sha256sum find sort comm ffprobe awk wc cp mv mkdir mktemp rmdir date rm; do
  require_command "$command_name"
done

annotation_dir="${data_root}/annotations"
video_dir="${data_root}/videos"
download_dir="${data_root}/.downloads"
manifest_dir="${data_root}/manifests"
archive_path="${download_dir}/tacos/${ARCHIVE_NAME}"
extract_dir=""
cleanup_extract() {
  if [[ -n "$extract_dir" && -d "$extract_dir" ]]; then
    rm -rf -- "$extract_dir"
  fi
}
trap cleanup_extract EXIT
mkdir -p "$annotation_dir" "$video_dir" "$download_dir" "$manifest_dir"

echo "[source] ${DATASET_PAGE}"
echo "[revision] ${DATASET_REVISION}"
hf download "$DATASET_REPO" \
  tacos/train.jsonl tacos/val.jsonl tacos/test.jsonl \
  --repo-type dataset --revision "$DATASET_REVISION" --local-dir "$download_dir"

expected_ids="${manifest_dir}/expected_video_ids.txt"
available_ids="${manifest_dir}/available_video_ids.txt"
missing_ids="${manifest_dir}/missing_video_ids.txt"
: > "$expected_ids"
for index in "${!SPLITS[@]}"; do
  split="${SPLITS[$index]}"
  source_path="${download_dir}/tacos/${split}.jsonl"
  destination="${annotation_dir}/${split}.jsonl"
  cp "$source_path" "${destination}.incomplete"
  mv "${destination}.incomplete" "$destination"
  actual_sha="$(sha256sum "$destination" | awk '{print $1}')"
  [[ "$actual_sha" == "${ANNOTATION_SHA256[$index]}" ]] || die "${split}.jsonl checksum mismatch"
  jq -e -s --argjson expected "${EXPECTED_QUERIES[$index]}" '
    length == $expected and all(.[];
      (.qid | type) == "string" and (.vid | type) == "string" and
      (.query | type) == "string" and (.duration | type) == "number" and
      (.relevant_windows | type) == "array" and
      all(.relevant_windows[]; type == "array" and length == 2))
  ' "$destination" >/dev/null || die "invalid ${split} annotation schema or row count"
  jq -r '.vid' "$destination" >> "$expected_ids"
  echo "[OK] ${split}: ${EXPECTED_QUERIES[$index]} queries"
done
sort -u -o "$expected_ids" "$expected_ids"
[[ "$(wc -l < "$expected_ids")" -eq "$EXPECTED_VIDEOS" ]] || die "annotations do not reference 127 videos"

find "$video_dir" -type f -iname '*.mp4' -printf '%f\n' \
  | awk '{sub(/[.][^.]+$/, ""); print}' | sort -u > "$available_ids"
if [[ "$(comm -12 "$expected_ids" "$available_ids" | wc -l)" -ne "$EXPECTED_VIDEOS" ]]; then
  if [[ -n "$(find "$video_dir" -type f -print -quit)" ]]; then
    die "video directory is partial or uses unnormalized names; remove ${video_dir} before retrying"
  fi
  hf download "$DATASET_REPO" "tacos/${ARCHIVE_NAME}" \
    --repo-type dataset --revision "$DATASET_REVISION" --local-dir "$download_dir"
  tar -tzf "$archive_path" > "${manifest_dir}/archive_members.txt"
  if awk '/(^|\/)[.][.](\/|$)/ || /^\// {bad=1} END {exit bad ? 0 : 1}' \
    "${manifest_dir}/archive_members.txt"; then
    die "unsafe path in TACoS archive"
  fi
  extract_dir="$(mktemp -d "${data_root}/.tacos-extract.XXXXXX")"
  echo "[extract] compressed archive into temporary staging"
  tar -xzf "$archive_path" -C "$extract_dir"
  archive_video_dir="${extract_dir}/videos_3fps_480_noaudio"
  [[ -d "$archive_video_dir" ]] || die "archive is missing videos_3fps_480_noaudio/"
  archive_video_count="$(find "$archive_video_dir" -maxdepth 1 -type f -iname '*.mp4' | wc -l)"
  [[ "$archive_video_count" -eq "$EXPECTED_ARCHIVE_VIDEOS" ]] || \
    die "expected ${EXPECTED_ARCHIVE_VIDEOS} archive videos, found ${archive_video_count}"

  selected_video_dir="${extract_dir}/selected"
  mkdir "$selected_video_dir"
  missing_archive_ids="${manifest_dir}/missing_archive_video_ids.txt"
  : > "$missing_archive_ids"
  while IFS= read -r video_id; do
    source_video="${archive_video_dir}/${video_id}-cam-002.mp4"
    if [[ ! -f "$source_video" ]]; then
      echo "$video_id" >> "$missing_archive_ids"
    fi
  done < "$expected_ids"
  [[ ! -s "$missing_archive_ids" ]] || \
    die "archive is missing referenced videos; see ${missing_archive_ids}"

  while IFS= read -r video_id; do
    mv "${archive_video_dir}/${video_id}-cam-002.mp4" "${selected_video_dir}/${video_id}.mp4"
  done < "$expected_ids"
  rmdir "$video_dir"
  mv "$selected_video_dir" "$video_dir"
  echo "[extract] selected and normalized ${EXPECTED_VIDEOS} videos into ${video_dir}"
  cleanup_extract
  extract_dir=""
fi

find "$video_dir" -type f -iname '*.mp4' -printf '%f\n' \
  | awk '{sub(/[.][^.]+$/, ""); print}' | sort -u > "$available_ids"
comm -23 "$expected_ids" "$available_ids" > "$missing_ids"
[[ ! -s "$missing_ids" ]] || die "dataset is incomplete; see ${missing_ids}"

video_count=0
audio_count=0
while IFS= read -r video; do
  video_count=$((video_count + 1))
  stream_info="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate \
    -of csv=p=0 "$video")" || die "invalid video: ${video}"
  IFS=, read -r width height frame_rate <<< "$stream_info"
  [[ "$width" =~ ^[0-9]+$ && "$height" =~ ^[0-9]+$ ]] || die "invalid dimensions: ${video}"
  ((width <= 854 && height <= 480)) || die "video exceeds 480p envelope: ${video} (${width}x${height})"
  awk -v rate="$frame_rate" 'BEGIN {
    split(rate, parts, "/"); fps = parts[2] ? parts[1] / parts[2] : parts[1]; exit !(fps > 0 && fps <= 3.01)
  }' || die "video exceeds 3 fps: ${video} (${frame_rate})"
  if [[ -n "$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$video")" ]]; then
    audio_count=$((audio_count + 1))
  fi
done < <(find "$video_dir" -type f -iname '*.mp4' | sort)
[[ "$video_count" -eq "$EXPECTED_VIDEOS" ]] || die "expected 127 videos, found ${video_count}"
[[ "$audio_count" -eq 0 ]] || die "expected no audio streams, found ${audio_count}"

if [[ "$keep_archive" == false && -f "$archive_path" ]]; then
  rm -f "$archive_path"
fi
jq -n \
  --arg source "$DATASET_PAGE" --arg revision "$DATASET_REVISION" \
  --arg prepared_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson videos "$video_count" \
  '{schema: 1, source: $source, revision: $revision, prepared_at: $prepared_at,
    videos: $videos, fps: 3, resolution: "480p", audio: false, compressed: true}' \
  > "${data_root}/dataset-manifest.json"

echo "[ready] ${data_root}"
echo "Set TACOS_ROOT=${data_root}"
