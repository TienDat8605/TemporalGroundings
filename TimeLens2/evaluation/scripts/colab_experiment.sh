#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
REMOTE_CONTROLLER="$SCRIPT_DIR/colab_remote_job.py"
REMOTE_SPEC_PATH="/content/timelens2_colab_job_spec.json"
STATE_ROOT="${COLAB_STATE_ROOT:-$WORKSPACE_ROOT/results/colab_runs}"

DEFAULT_SESSION="${COLAB_SESSION:-timelens2}"
DEFAULT_WORKDIR="${COLAB_WORKDIR:-evaluation}"
DEFAULT_SETUP_COMMAND="${COLAB_SETUP_COMMAND:-python -m pip install -e .}"
DEFAULT_POLL_SECONDS="${COLAB_POLL_SECONDS:-10}"
DEFAULT_CHECKPOINT_SECONDS="${COLAB_CHECKPOINT_SECONDS:-30}"
DEFAULT_LOST_AFTER_FAILURES="${COLAB_LOST_AFTER_FAILURES:-6}"

usage() {
  cat <<'EOF'
Usage:
  evaluation/scripts/colab_experiment.sh start --dataset-command DOWNLOAD --command RUN [options]
  evaluation/scripts/colab_experiment.sh status [--session NAME]
  evaluation/scripts/colab_experiment.sh logs [--session NAME] [--follow] [--lines N]
  evaluation/scripts/colab_experiment.sh monitor [--session NAME]
  evaluation/scripts/colab_experiment.sh fetch [--session NAME] [--destination FILE]
  evaluation/scripts/colab_experiment.sh cancel [--session NAME]
  evaluation/scripts/colab_experiment.sh stop [--session NAME]

start options:
  --command CMD          Required experiment command, run inside --workdir.
  --setup-command CMD    Setup command run before the experiment.
  --no-setup             Skip dependency installation.
  --session NAME         Colab session name (default: timelens2).
  --dataset-command CMD  Required remote dataset download/extraction command.
  --no-dataset-download  Skip the dataset phase for data-free tests.
  --workdir PATH         Repository-relative working directory (default: evaluation).
  --env-file FILE        Upload KEY=VALUE entries separately from source.
  --youtube-cookies FILE Upload a Netscape-format cookies file for yt-dlp.
                         The remote copy is deleted when the job finishes.
  --resume-checkpoint FILE
                         Restore a previously downloaded OMTG checkpoint.
  --sync-path PATH       Additional repository-relative path to upload; repeatable.
                         Always includes evaluation, README.md, LICENSE, .gitignore.
  --output-path PATH     Additional repository-relative result path for fetch; repeatable.
                         Always includes evaluation/outputs.
  --detach               Return after launch instead of monitoring logs.
  --poll-seconds N       Monitor interval (default: 10).

Example:
  evaluation/scripts/colab_experiment.sh start \
    --env-file .env.colab \
    --dataset-command 'bash scripts/download_vue_tr_v2.sh' \
    --command 'MODELS="TimeLens2-4B" DATASETS="VUE_TR_V2_1fps_limit_64_px336_ctx8k_t4" N_GPU=1 bash scripts/srun_eval_all/run_grounding.sh'

Notes:
  - start always archives and uploads the current files, including uncommitted edits.
  - Sessions always use one T4 GPU; Google Drive is not mounted.
  - .env files, datasets, caches, checkpoints, and outputs are excluded from source sync.
  - Dataset commands run remotely with TIMELENS2_DATA_ROOT=/content/timelens2-data.
  - Use --env-file for API credentials; values are not printed.
  - YouTube may require browser cookies for Colab IPs. Cookies are credentials:
    keep the file private and use a disposable YouTube account if possible.
  - Ctrl-C while monitoring detaches. It does not cancel the remote job.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_session() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || die "unsafe session name: $1"
}

session_accepts_files() {
  local session="$1"
  local attempt
  for attempt in 1 2 3; do
    if colab upload -s "$session" "$REMOTE_CONTROLLER" \
      /content/.timelens2_session_probe.py >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

state_file_for() {
  printf '%s/%s/current.json\n' "$STATE_ROOT" "$1"
}

load_state() {
  local session="$1"
  local state_file
  state_file="$(state_file_for "$session")"
  [[ -f "$state_file" ]] || die "no local job state for session '$session'; run start first"
  printf '%s\n' "$state_file"
}

ensure_session() {
  local session="$1"
  local status_output
  status_output="$(colab status -s "$session" 2>&1 || true)"
  if [[ "$status_output" == *"[$session] "* ]]; then
    if session_accepts_files "$session"; then
      printf 'Using active Colab session: %s\n' "$session"
      return
    fi
    printf 'Colab session %s is stale; recreating it...\n' "$session"
    colab stop -s "$session" >/dev/null 2>&1 || true
  fi
  printf 'Creating Colab session %s with GPU T4...\n' "$session"
  colab new -s "$session" --gpu T4
  status_output="$(colab status -s "$session" 2>&1 || true)"
  if [[ "$status_output" != *"[$session] "* ]]; then
    printf '%s\n' "$status_output" >&2
    die "Colab created session '$session' but it is not available in local session state"
  fi
  session_accepts_files "$session" \
    || die "Colab session '$session' was created but its file API is unavailable"
}

git_revision() {
  git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || printf 'nogit\n'
}

git_dirty_json() {
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null || true)" ]]; then
    printf 'true\n'
  else
    printf 'false\n'
  fi
}

validate_relative_repo_path() {
  local relative="$1"
  [[ -n "$relative" ]] || die 'empty repository-relative path'
  [[ "$relative" != /* ]] || die "path must be repository-relative: $relative"
  [[ "$relative" != '..' && "$relative" != ../* && "$relative" != */../* && "$relative" != */.. ]] \
    || die "path escapes repository: $relative"
}

build_source_archive() {
  local archive="$1"
  shift
  local sync_paths=("$@")
  local path
  for path in "${sync_paths[@]}"; do
    validate_relative_repo_path "$path"
    [[ -e "$REPO_ROOT/$path" ]] || die "sync path does not exist: $path"
  done

  tar \
    --exclude='.git' \
    --exclude='.colab-runs' \
    --exclude='*/__pycache__' \
    --exclude='*/.pytest_cache' \
    --exclude='*/.mypy_cache' \
    --exclude='*/.cache' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='*/.env' \
    --exclude='*/.env.*' \
    --exclude='cookies.txt' \
    --exclude='*/cookies.txt' \
    --exclude='*youtube*cookies*' \
    --exclude='*Youtube*Cookies*' \
    --exclude='data' \
    --exclude='*/data' \
    --exclude='outputs' \
    --exclude='*/outputs' \
    --exclude='output' \
    --exclude='*/output' \
    --exclude='*/work_dir' \
    --exclude='*/wandb' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    -czf "$archive" \
    -C "$REPO_ROOT" \
    "${sync_paths[@]}"
}

write_action_spec() {
  local destination="$1"
  local action="$2"
  local job_id="$3"
  jq -n \
    --arg action "$action" \
    --arg job_id "$job_id" \
    '{action: $action, job_id: $job_id}' >"$destination"
}

download_job_file() {
  local session="$1"
  local remote_path="$2"
  local local_path="$3"
  mkdir -p "$(dirname "$local_path")"
  colab download -s "$session" "$remote_path" "$local_path"
}

sync_job_log() {
  local session="$1"
  local remote_log="$2"
  local local_log="$3"
  local offset_name="$4"
  local -n offset_ref="$offset_name"

  download_job_file "$session" "$remote_log" "$local_log" >/dev/null 2>&1 || return 1
  local size
  size="$(wc -c <"$local_log")"
  if (( size > offset_ref )); then
    tail -c "+$((offset_ref + 1))" "$local_log"
  elif (( size < offset_ref )); then
    cat "$local_log"
  fi
  offset_ref="$size"
}

sync_job_checkpoint() {
  local session="$1"
  local remote_checkpoint="$2"
  local local_checkpoint="$3"
  local temporary_checkpoint="${local_checkpoint}.part"

  mkdir -p "$(dirname "$local_checkpoint")"
  rm -f -- "$temporary_checkpoint"
  if ! download_job_file \
    "$session" "$remote_checkpoint" "$temporary_checkpoint" >/dev/null 2>&1; then
    rm -f -- "$temporary_checkpoint"
    return 1
  fi
  mv -f "$temporary_checkpoint" "$local_checkpoint"
}

mark_local_job_lost() {
  local local_status="$1"
  local message="$2"
  local temporary_status="${local_status}.tmp"
  local updated_at
  updated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$(dirname "$local_status")"
  if [[ -f "$local_status" ]]; then
    jq \
      --arg message "$message" \
      --arg updated_at "$updated_at" \
      '.state = "LOST" | .message = $message | .updated_at = $updated_at' \
      "$local_status" >"$temporary_status"
  else
    jq -n \
      --arg message "$message" \
      --arg updated_at "$updated_at" \
      '{state: "LOST", message: $message, updated_at: $updated_at}' \
      >"$temporary_status"
  fi
  mv -f "$temporary_status" "$local_status"
}

show_status() {
  local session="$1"
  local state_file
  state_file="$(load_state "$session")"
  local remote_status local_status
  remote_status="$(jq -r '.remote_status_path' "$state_file")"
  local_status="$(jq -r '.local_status_path' "$state_file")"

  colab status -s "$session"
  download_job_file "$session" "$remote_status" "$local_status"
  jq . "$local_status"
}

monitor_job() {
  local session="$1"
  local poll_seconds="$2"
  local initial_lines="${3:-0}"
  local state_file
  state_file="$(load_state "$session")"
  local remote_log remote_status remote_checkpoint local_log local_status local_checkpoint
  remote_log="$(jq -r '.remote_log_path' "$state_file")"
  remote_status="$(jq -r '.remote_status_path' "$state_file")"
  remote_checkpoint="$(
    jq -r '.remote_checkpoint_path // (.remote_run_dir + "/omtg_checkpoint.tar.gz")' "$state_file"
  )"
  local_log="$(jq -r '.local_log_path' "$state_file")"
  local_status="$(jq -r '.local_status_path' "$state_file")"
  local_checkpoint="$(
    jq -r '.local_checkpoint_path // (.local_run_dir + "/omtg_checkpoint.tar.gz")' "$state_file"
  )"
  local offset=0
  local failures=0
  local checkpoint_elapsed="$DEFAULT_CHECKPOINT_SECONDS"

  mkdir -p "$(dirname "$local_log")"
  if [[ "$initial_lines" -gt 0 ]] && download_job_file "$session" "$remote_log" "$local_log" >/dev/null 2>&1; then
    tail -n "$initial_lines" "$local_log"
    offset="$(wc -c <"$local_log")"
  fi

  trap 'printf "\nDetached; the Colab job is still running.\n"; return 130' INT
  while true; do
    local retrieved_any=false
    if sync_job_log "$session" "$remote_log" "$local_log" offset; then
      retrieved_any=true
    fi

    if download_job_file "$session" "$remote_status" "$local_status" >/dev/null 2>&1; then
      retrieved_any=true
      local state
      state="$(jq -r '.state // "UNKNOWN"' "$local_status")"
      case "$state" in
        SUCCEEDED)
          sync_job_log "$session" "$remote_log" "$local_log" offset || true
          sync_job_checkpoint \
            "$session" "$remote_checkpoint" "$local_checkpoint" || true
          download_job_file "$session" "$remote_status" "$local_status" >/dev/null 2>&1 || true
          printf '\nRemote job succeeded.\n'
          printf 'Local log: %s\nLocal status: %s\n' "$local_log" "$local_status"
          [[ -f "$local_checkpoint" ]] \
            && printf 'Local checkpoint: %s\n' "$local_checkpoint"
          trap - INT
          return 0
          ;;
        FAILED_SETUP|FAILED_DATASET|FAILED|CANCELED)
          sync_job_log "$session" "$remote_log" "$local_log" offset || true
          sync_job_checkpoint \
            "$session" "$remote_checkpoint" "$local_checkpoint" || true
          download_job_file "$session" "$remote_status" "$local_status" >/dev/null 2>&1 || true
          printf '\nRemote job ended with state: %s\n' "$state" >&2
          jq . "$local_status" >&2
          printf 'Local log: %s\nLocal status: %s\n' "$local_log" "$local_status" >&2
          [[ -f "$local_checkpoint" ]] \
            && printf 'Local checkpoint: %s\n' "$local_checkpoint" >&2
          trap - INT
          return 1
          ;;
      esac
    fi

    if (( checkpoint_elapsed >= DEFAULT_CHECKPOINT_SECONDS )); then
      if sync_job_checkpoint "$session" "$remote_checkpoint" "$local_checkpoint"; then
        retrieved_any=true
      fi
      checkpoint_elapsed=0
    fi

    if [[ "$retrieved_any" == true ]]; then
      failures=0
    else
      failures=$((failures + 1))
    fi

    if (( failures == 3 )); then
      printf 'Remote job files unavailable for three checks; checking Colab session...\n' >&2
      colab status -s "$session" || true
    fi
    if (( failures >= DEFAULT_LOST_AFTER_FAILURES )); then
      local lost_message
      lost_message="Remote job files were unavailable for $failures consecutive checks; the Colab VM was likely recycled."
      mark_local_job_lost "$local_status" "$lost_message"
      printf '\nRemote job state is LOST: %s\n' "$lost_message" >&2
      printf 'Local log: %s\nLocal status: %s\n' "$local_log" "$local_status" >&2
      if [[ -f "$local_checkpoint" ]]; then
        printf 'Resume checkpoint: %s\n' "$local_checkpoint" >&2
      else
        printf 'No local checkpoint was downloaded; this run must restart from the beginning.\n' >&2
      fi
      trap - INT
      return 1
    fi
    sleep "$poll_seconds"
    checkpoint_elapsed=$((checkpoint_elapsed + poll_seconds))
  done
}

start_job() {
  local session="$DEFAULT_SESSION"
  local workdir="$DEFAULT_WORKDIR"
  local setup_command="$DEFAULT_SETUP_COMMAND"
  local dataset_command=""
  local require_dataset_command=true
  local run_command=""
  local env_file=""
  local youtube_cookies=""
  local resume_checkpoint=""
  local detach=false
  local poll_seconds="$DEFAULT_POLL_SECONDS"
  local sync_paths=(evaluation README.md LICENSE .gitignore)
  local output_paths=(evaluation/outputs)

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session) session="${2:?missing value for --session}"; shift 2 ;;
      --workdir) workdir="${2:?missing value for --workdir}"; shift 2 ;;
      --setup-command) setup_command="${2:?missing value for --setup-command}"; shift 2 ;;
      --no-setup) setup_command=""; shift ;;
      --dataset-command) dataset_command="${2:?missing value for --dataset-command}"; shift 2 ;;
      --no-dataset-download) require_dataset_command=false; shift ;;
      --command) run_command="${2:?missing value for --command}"; shift 2 ;;
      --env-file) env_file="${2:?missing value for --env-file}"; shift 2 ;;
      --youtube-cookies) youtube_cookies="${2:?missing value for --youtube-cookies}"; shift 2 ;;
      --resume-checkpoint) resume_checkpoint="${2:?missing value for --resume-checkpoint}"; shift 2 ;;
      --detach) detach=true; shift ;;
      --poll-seconds) poll_seconds="${2:?missing value for --poll-seconds}"; shift 2 ;;
      --sync-path) sync_paths+=("${2:?missing value for --sync-path}"); shift 2 ;;
      --output-path) output_paths+=("${2:?missing value for --output-path}"); shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown start option: $1" ;;
    esac
  done

  [[ -n "$run_command" ]] || die 'start requires --command CMD'
  if [[ "$require_dataset_command" == true && -z "$dataset_command" ]]; then
    die 'start requires --dataset-command CMD (or --no-dataset-download for data-free tests)'
  fi
  if [[ "$dataset_command" == *"OWNER/DATASET"* || "$dataset_command" == *"OWNER/VUE_TR_V2"* ]]; then
    die 'replace the placeholder OWNER/DATASET in --dataset-command with the real dataset source'
  fi
  validate_session "$session"
  validate_relative_repo_path "$workdir"
  [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || die '--poll-seconds must be a positive integer'
  if [[ -n "$env_file" ]]; then
    [[ -f "$env_file" ]] || die "environment file not found: $env_file"
  fi
  if [[ -n "$youtube_cookies" ]]; then
    [[ -f "$youtube_cookies" ]] || die "YouTube cookies file not found: $youtube_cookies"
    if ! head -n 2 "$youtube_cookies" | grep -Eq '^# (Netscape )?HTTP Cookie File'; then
      die 'YouTube cookies must use Mozilla/Netscape cookie-file format'
    fi
  fi
  if [[ -n "$resume_checkpoint" ]]; then
    [[ -f "$resume_checkpoint" ]] || die "resume checkpoint not found: $resume_checkpoint"
    tar -tzf "$resume_checkpoint" >/dev/null \
      || die "resume checkpoint is not a readable gzip tar archive: $resume_checkpoint"
    local checkpoint_member
    while IFS= read -r checkpoint_member; do
      [[ "$checkpoint_member" != /* ]] \
        || die "resume checkpoint contains an absolute path: $checkpoint_member"
      [[ "/$checkpoint_member/" != *"/../"* ]] \
        || die "resume checkpoint contains a parent traversal: $checkpoint_member"
    done < <(tar -tzf "$resume_checkpoint")
  fi
  local output_path
  for output_path in "${output_paths[@]}"; do
    validate_relative_repo_path "$output_path"
  done

  ensure_session "$session"

  local temporary_dir archive revision dirty archive_hash job_id
  temporary_dir="$(mktemp -d)"
  archive="$temporary_dir/source.tar.gz"
  revision="$(git_revision)"
  dirty="$(git_dirty_json)"
  printf 'Packaging current source tree...\n'
  build_source_archive "$archive" "${sync_paths[@]}"
  archive_hash="$(sha256sum "$archive" | awk '{print $1}')"
  job_id="$(date -u +%Y%m%dT%H%M%SZ)-${revision}-${archive_hash:0:10}-${BASHPID}"

  local remote_archive remote_env remote_youtube_cookies remote_resume_checkpoint
  local remote_run_dir remote_status remote_log remote_checkpoint remote_output_archive
  remote_archive="/content/timelens2_${job_id}_source.tar.gz"
  remote_env=""
  if [[ -n "$env_file" ]]; then
    remote_env="/content/timelens2_${job_id}.env"
  fi
  remote_youtube_cookies=""
  if [[ -n "$youtube_cookies" ]]; then
    remote_youtube_cookies="/content/timelens2_${job_id}_youtube_cookies.txt"
  fi
  remote_resume_checkpoint=""
  if [[ -n "$resume_checkpoint" ]]; then
    remote_resume_checkpoint="/content/timelens2_${job_id}_resume.tar.gz"
  fi
  remote_run_dir="/content/timelens2-runs/$job_id"
  remote_status="$remote_run_dir/status.json"
  remote_log="$remote_run_dir/job.log"
  remote_checkpoint="$remote_run_dir/omtg_checkpoint.tar.gz"
  remote_output_archive="/content/timelens2_${job_id}_outputs.tar.gz"

  local output_paths_json spec_file resume_checkpoint_hash
  output_paths_json="$(printf '%s\n' "${output_paths[@]}" | jq -R . | jq -s .)"
  resume_checkpoint_hash=""
  if [[ -n "$resume_checkpoint" ]]; then
    resume_checkpoint_hash="$(sha256sum "$resume_checkpoint" | awk '{print $1}')"
  fi
  spec_file="$temporary_dir/job_spec.json"
  jq -n \
    --arg action launch \
    --arg job_id "$job_id" \
    --arg archive_path "$remote_archive" \
    --arg archive_sha256 "$archive_hash" \
    --arg source_revision "$revision" \
    --argjson source_dirty "$dirty" \
    --arg workdir "$workdir" \
    --arg setup_command "$setup_command" \
    --arg dataset_command "$dataset_command" \
    --arg run_command "$run_command" \
    --arg env_file_path "$remote_env" \
    --arg youtube_cookies_path "$remote_youtube_cookies" \
    --arg resume_checkpoint_path "$remote_resume_checkpoint" \
    --arg resume_checkpoint_sha256 "$resume_checkpoint_hash" \
    --argjson output_paths "$output_paths_json" \
    '{
      action: $action,
      job_id: $job_id,
      archive_path: $archive_path,
      archive_sha256: $archive_sha256,
      source_revision: $source_revision,
      source_dirty: $source_dirty,
      workdir: $workdir,
      setup_command: $setup_command,
      dataset_command: $dataset_command,
      run_command: $run_command,
      env_file_path: (if $env_file_path == "" then null else $env_file_path end),
      youtube_cookies_path: (if $youtube_cookies_path == "" then null else $youtube_cookies_path end),
      resume_checkpoint_path: (if $resume_checkpoint_path == "" then null else $resume_checkpoint_path end),
      resume_checkpoint_sha256: (if $resume_checkpoint_sha256 == "" then null else $resume_checkpoint_sha256 end),
      output_paths: $output_paths
    }' >"$spec_file"

  printf 'Uploading source archive (%s)...\n' "$(du -h "$archive" | awk '{print $1}')"
  colab upload -s "$session" "$archive" "$remote_archive"
  if [[ -n "$env_file" ]]; then
    colab upload -s "$session" "$env_file" "$remote_env"
  fi
  if [[ -n "$youtube_cookies" ]]; then
    colab upload -s "$session" "$youtube_cookies" "$remote_youtube_cookies"
  fi
  if [[ -n "$resume_checkpoint" ]]; then
    printf 'Uploading resume checkpoint (%s)...\n' "$(du -h "$resume_checkpoint" | awk '{print $1}')"
    colab upload -s "$session" "$resume_checkpoint" "$remote_resume_checkpoint"
  fi
  colab upload -s "$session" "$spec_file" "$REMOTE_SPEC_PATH"
  printf 'Launching remote job %s...\n' "$job_id"
  colab exec -s "$session" -f "$REMOTE_CONTROLLER" --timeout 120
  local launch_status
  launch_status="$temporary_dir/launch_status.json"
  if ! colab download -s "$session" "$remote_status" "$launch_status"; then
    die "remote controller did not create job state for '$job_id'; inspect the colab exec output above"
  fi
  if [[ "$(jq -r '.job_id // ""' "$launch_status")" != "$job_id" ]]; then
    die "remote controller returned state for the wrong job id"
  fi

  local local_run_dir state_file
  local_run_dir="$STATE_ROOT/$session/$job_id"
  state_file="$(state_file_for "$session")"
  mkdir -p "$local_run_dir" "$(dirname "$state_file")"
  jq -n \
    --arg session "$session" \
    --arg job_id "$job_id" \
    --arg remote_run_dir "$remote_run_dir" \
    --arg remote_status_path "$remote_status" \
    --arg remote_log_path "$remote_log" \
    --arg remote_checkpoint_path "$remote_checkpoint" \
    --arg remote_output_archive "$remote_output_archive" \
    --arg local_run_dir "$local_run_dir" \
    --arg local_status_path "$local_run_dir/status.json" \
    --arg local_log_path "$local_run_dir/job.log" \
    --arg local_checkpoint_path "$local_run_dir/omtg_checkpoint.tar.gz" \
    '{
      session: $session,
      job_id: $job_id,
      remote_run_dir: $remote_run_dir,
      remote_status_path: $remote_status_path,
      remote_log_path: $remote_log_path,
      remote_checkpoint_path: $remote_checkpoint_path,
      remote_output_archive: $remote_output_archive,
      local_run_dir: $local_run_dir,
      local_status_path: $local_status_path,
      local_log_path: $local_log_path,
      local_checkpoint_path: $local_checkpoint_path
    }' >"$state_file"
  cp "$spec_file" "$local_run_dir/job_spec.json"
  rm -rf -- "$temporary_dir"

  printf 'Remote job launched. Session=%s Job=%s\n' "$session" "$job_id"
  if [[ "$detach" == true ]]; then
    printf 'Monitor with: %s monitor --session %s\n' "$0" "$session"
  else
    monitor_job "$session" "$poll_seconds" 0
  fi
}

logs_command() {
  local session="$DEFAULT_SESSION"
  local follow=false
  local lines=200
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session) session="${2:?missing value for --session}"; shift 2 ;;
      --follow|-f) follow=true; shift ;;
      --lines|-n) lines="${2:?missing value for --lines}"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown logs option: $1" ;;
    esac
  done
  validate_session "$session"
  [[ "$lines" =~ ^[0-9]+$ ]] || die '--lines must be a non-negative integer'
  if [[ "$follow" == true ]]; then
    monitor_job "$session" "$DEFAULT_POLL_SECONDS" "$lines"
    return
  fi
  local state_file remote_log local_log
  state_file="$(load_state "$session")"
  remote_log="$(jq -r '.remote_log_path' "$state_file")"
  local_log="$(jq -r '.local_log_path' "$state_file")"
  download_job_file "$session" "$remote_log" "$local_log"
  if [[ "$lines" -eq 0 ]]; then
    cat "$local_log"
  else
    tail -n "$lines" "$local_log"
  fi
}

simple_session_arg() {
  local session="$DEFAULT_SESSION"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session) session="${2:?missing value for --session}"; shift 2 ;;
      -h|--help) usage; return 2 ;;
      *) die "unknown option: $1" ;;
    esac
  done
  validate_session "$session"
  printf '%s\n' "$session"
}

controller_action() {
  local session="$1"
  local action="$2"
  local state_file job_id temporary_dir spec_file
  state_file="$(load_state "$session")"
  job_id="$(jq -r '.job_id' "$state_file")"
  temporary_dir="$(mktemp -d)"
  spec_file="$temporary_dir/action.json"
  write_action_spec "$spec_file" "$action" "$job_id"
  colab upload -s "$session" "$spec_file" "$REMOTE_SPEC_PATH"
  colab exec -s "$session" -f "$REMOTE_CONTROLLER" --timeout 120
  rm -rf -- "$temporary_dir"
}

fetch_command() {
  local session="$DEFAULT_SESSION"
  local destination=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session) session="${2:?missing value for --session}"; shift 2 ;;
      --destination) destination="${2:?missing value for --destination}"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown fetch option: $1" ;;
    esac
  done
  validate_session "$session"
  local state_file local_run_dir remote_archive
  state_file="$(load_state "$session")"
  local_run_dir="$(jq -r '.local_run_dir' "$state_file")"
  remote_archive="$(jq -r '.remote_output_archive' "$state_file")"
  if [[ -z "$destination" ]]; then
    destination="$local_run_dir/outputs.tar.gz"
  fi
  controller_action "$session" archive
  download_job_file "$session" "$remote_archive" "$destination"
  printf 'Downloaded outputs to %s\n' "$destination"
}

main() {
  require_command colab
  require_command jq
  require_command tar
  require_command sha256sum
  [[ "$DEFAULT_CHECKPOINT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || die 'COLAB_CHECKPOINT_SECONDS must be a positive integer'
  [[ "$DEFAULT_LOST_AFTER_FAILURES" =~ ^[1-9][0-9]*$ ]] \
    || die 'COLAB_LOST_AFTER_FAILURES must be a positive integer'
  [[ -f "$REMOTE_CONTROLLER" ]] || die "remote controller missing: $REMOTE_CONTROLLER"

  local command="${1:-help}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
  case "$command" in
    start|run) start_job "$@" ;;
    status)
      local session
      session="$(simple_session_arg "$@")" || return $?
      show_status "$session"
      ;;
    logs) logs_command "$@" ;;
    monitor)
      local session
      session="$(simple_session_arg "$@")" || return $?
      monitor_job "$session" "$DEFAULT_POLL_SECONDS" 200
      ;;
    fetch) fetch_command "$@" ;;
    cancel)
      local session
      session="$(simple_session_arg "$@")" || return $?
      controller_action "$session" cancel
      ;;
    stop)
      local session
      session="$(simple_session_arg "$@")" || return $?
      colab stop -s "$session"
      ;;
    help|-h|--help) usage ;;
    *) die "unknown command: $command" ;;
  esac
}

main "$@"
