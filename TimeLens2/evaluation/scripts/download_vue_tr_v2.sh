#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${TIMELENS2_DATA_ROOT:-/content/timelens2-data}"
DATASET_ROOT="${VUE_TR_V2_ROOT:-$DATA_ROOT/VUE_TR_V2}"
VIDI_SOURCE_ROOT="$DATA_ROOT/.sources/vidi"
MAX_VIDEOS="${VUE_TR_MAX_VIDEOS:-0}"
YTDLP_SLEEP_INTERVAL="${YTDLP_SLEEP_INTERVAL:-5}"
YTDLP_MAX_SLEEP_INTERVAL="${YTDLP_MAX_SLEEP_INTERVAL:-10}"
YTDLP_USE_COOKIES="${YTDLP_USE_COOKIES:-true}"
YTDLP_ENABLE_POT_PROVIDER="${YTDLP_ENABLE_POT_PROVIDER:-false}"
YTDLP_POT_PROVIDER_VERSION="${YTDLP_POT_PROVIDER_VERSION:-1.3.1}"

[[ "$MAX_VIDEOS" =~ ^[0-9]+$ ]] || {
  printf 'VUE_TR_MAX_VIDEOS must be a non-negative integer\n' >&2
  exit 2
}
[[ "$YTDLP_SLEEP_INTERVAL" =~ ^[0-9]+$ && "$YTDLP_MAX_SLEEP_INTERVAL" =~ ^[0-9]+$ ]] || {
  printf 'YTDLP_SLEEP_INTERVAL and YTDLP_MAX_SLEEP_INTERVAL must be non-negative integers\n' >&2
  exit 2
}
(( YTDLP_MAX_SLEEP_INTERVAL >= YTDLP_SLEEP_INTERVAL )) || {
  printf 'YTDLP_MAX_SLEEP_INTERVAL must be at least YTDLP_SLEEP_INTERVAL\n' >&2
  exit 2
}
[[ "$YTDLP_USE_COOKIES" =~ ^(true|false)$ ]] || {
  printf 'YTDLP_USE_COOKIES must be true or false\n' >&2
  exit 2
}
[[ "$YTDLP_ENABLE_POT_PROVIDER" =~ ^(true|false)$ ]] || {
  printf 'YTDLP_ENABLE_POT_PROVIDER must be true or false\n' >&2
  exit 2
}

mkdir -p "$DATA_ROOT/.sources" "$DATASET_ROOT/videos"

if [[ ! -d "$VIDI_SOURCE_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/bytedance/vidi.git "$VIDI_SOURCE_ROOT"
fi

cp "$VIDI_SOURCE_ROOT/VUE_TR_V2/VUE-TRv2_ground_truth.json" "$DATASET_ROOT/VUE-TRv2_ground_truth.json"
cp "$VIDI_SOURCE_ROOT/VUE_TR_V2/video_id.txt" "$DATASET_ROOT/video_id.txt"

python -m pip install --upgrade 'yt-dlp[default]'

DENO_INSTALL_ROOT="${DENO_INSTALL:-$DATA_ROOT/.tools/deno}"
if ! command -v deno >/dev/null 2>&1 && [[ ! -x "$DENO_INSTALL_ROOT/bin/deno" ]]; then
  mkdir -p "$DENO_INSTALL_ROOT"
  curl -fsSL https://deno.land/install.sh | DENO_INSTALL="$DENO_INSTALL_ROOT" sh
fi
export PATH="$DENO_INSTALL_ROOT/bin:$PATH"
command -v deno >/dev/null 2>&1 || {
  printf 'Deno installation failed; yt-dlp requires a JavaScript runtime for YouTube.\n' >&2
  exit 1
}

pot_extractor_args=()
if [[ "$YTDLP_ENABLE_POT_PROVIDER" == true ]]; then
  POT_PROVIDER_ROOT="$DATA_ROOT/.tools/bgutil-ytdlp-pot-provider"
  python -m pip install --upgrade \
    "bgutil-ytdlp-pot-provider==$YTDLP_POT_PROVIDER_VERSION"
  if [[ ! -d "$POT_PROVIDER_ROOT/.git" ]]; then
    git clone --depth 1 --single-branch \
      --branch "$YTDLP_POT_PROVIDER_VERSION" \
      https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git \
      "$POT_PROVIDER_ROOT"
  fi
  if [[ ! -f "$POT_PROVIDER_ROOT/server/.deno-ready-$YTDLP_POT_PROVIDER_VERSION" ]]; then
    (
      cd "$POT_PROVIDER_ROOT/server"
      deno install --allow-scripts=npm:canvas --frozen
      touch ".deno-ready-$YTDLP_POT_PROVIDER_VERSION"
    )
  fi
  pot_extractor_args=(
    --extractor-args
    "youtubepot-bgutilscript:server_home=$POT_PROVIDER_ROOT/server"
    --extractor-args
    "youtube:player_client=mweb"
  )
  printf 'Using BgUtils PO-token provider %s for YouTube requests.\n' \
    "$YTDLP_POT_PROVIDER_VERSION"
fi

url_list="$(mktemp)"
if [[ "$MAX_VIDEOS" -gt 0 ]]; then
  head -n "$MAX_VIDEOS" "$DATASET_ROOT/video_id.txt" \
    | awk 'NF {print "https://www.youtube.com/watch?v=" $1}' > "$url_list"
else
  awk 'NF {print "https://www.youtube.com/watch?v=" $1}' \
    "$DATASET_ROOT/video_id.txt" > "$url_list"
fi

printf 'Downloading VUE-TR-V2 videos into %s\n' "$DATASET_ROOT/videos"
yt_dlp_args=(
  --js-runtimes deno
  --continue
  --no-overwrites
  --concurrent-fragments 4
  --sleep-requests 1
  --sleep-interval "$YTDLP_SLEEP_INTERVAL"
  --max-sleep-interval "$YTDLP_MAX_SLEEP_INTERVAL"
  --merge-output-format mp4
  --remux-video mp4
  -S 'res:480,ext:mp4:m4a'
  "${pot_extractor_args[@]}"
)
if [[ -n "${YTDLP_COOKIES_FILE:-}" && "$YTDLP_USE_COOKIES" == true ]]; then
  [[ -r "$YTDLP_COOKIES_FILE" ]] || {
    printf 'YTDLP_COOKIES_FILE is not readable: %s\n' "$YTDLP_COOKIES_FILE" >&2
    exit 2
  }
  printf 'Using uploaded YouTube cookies for yt-dlp authentication.\n'
  yt_dlp_args+=(--cookies "$YTDLP_COOKIES_FILE")
elif [[ -n "${YTDLP_COOKIES_FILE:-}" ]]; then
  printf 'Uploaded cookies are not used for this public dataset; using anonymous PO-token requests.\n'
fi

probe_url="$(head -n 1 "$url_list")"
[[ -n "$probe_url" ]] || {
  printf 'The VUE-TR-V2 URL list is empty.\n' >&2
  exit 1
}
printf 'Checking YouTube access with one video before starting the batch...\n'
if ! yt-dlp "${yt_dlp_args[@]}" --simulate "$probe_url"; then
  printf 'YouTube preflight failed; the batch download was not attempted.\n' >&2
  exit 1
fi

download_code=0
yt-dlp \
  "${yt_dlp_args[@]}" \
  --batch-file "$url_list" \
  --ignore-errors \
  -o "$DATASET_ROOT/videos/%(id)s.%(ext)s" \
  || download_code=$?

video_count="$(find "$DATASET_ROOT/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
if [[ "$video_count" -eq 0 ]]; then
  printf 'No VUE-TR-V2 videos were downloaded (yt-dlp exit code %s).\n' "$download_code" >&2
  printf 'The PO-token preflight passed, but no MP4 files were produced; inspect the yt-dlp errors above.\n' >&2
  exit 1
fi

printf 'VUE-TR-V2 ready: %s MP4 videos, annotations at %s\n' "$video_count" "$DATASET_ROOT"
if [[ "$download_code" -ne 0 ]]; then
  printf 'Warning: yt-dlp reported unavailable videos; evaluation will use the downloaded subset.\n' >&2
fi
