#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 {omtg|tacos|charades|activitynet} OUTPUT_DIR [runner arguments...]" >&2
  exit 2
fi

benchmark="$1"
output_dir="$2"
shift 2
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$benchmark" in
  omtg) runner="$script_dir/run_omtg_local.sh" ;;
  tacos) runner="$script_dir/run_tacos_local.sh" ;;
  charades) runner="$script_dir/run_charades_local.sh" ;;
  activitynet) runner="$script_dir/run_activitynet_local.sh" ;;
  *) echo "unknown benchmark: $benchmark" >&2; exit 2 ;;
esac

mkdir -p "$output_dir"

"$runner" --spatial-policy dense --output "$output_dir/dense.jsonl" "$@"

for ratio in 0.0625 0.125 0.25; do
  ratio_name="${ratio/./p}"
  for policy in uniform semvid tpsa_query tpsa_motion tpsa_boundary; do
    "$runner" \
      --spatial-policy "$policy" \
      --retention-ratio "$ratio" \
      --output "$output_dir/${policy}-${ratio_name}.jsonl" \
      "$@"
  done
done
