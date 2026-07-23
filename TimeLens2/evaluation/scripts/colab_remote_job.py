#!/usr/bin/env python3
"""Remote background-job controller used by colab_experiment.sh."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


REMOTE_CONTENT_ROOT = Path(os.environ.get("TIMELENS2_COLAB_CONTENT_ROOT", "/content"))
DEFAULT_SPEC_PATH = REMOTE_CONTENT_ROOT / "timelens2_colab_job_spec.json"
REMOTE_RUN_ROOT = REMOTE_CONTENT_ROOT / "timelens2-runs"
TERMINAL_STATES = {"SUCCEEDED", "FAILED_SETUP", "FAILED_DATASET", "FAILED", "CANCELED"}
SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UPLOADED_FILE_KEYS = (
    "env_file_path",
    "youtube_cookies_path",
    "resume_checkpoint_path",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_job_id(job_id: str) -> str:
    if not SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError(f"Unsafe job id: {job_id!r}")
    return job_id


def job_paths(job_id: str) -> dict[str, Path]:
    validate_job_id(job_id)
    run_dir = REMOTE_RUN_ROOT / job_id
    return {
        "run_dir": run_dir,
        "repo": run_dir / "repo",
        "spec": run_dir / "job_spec.json",
        "status": run_dir / "status.json",
        "log": run_dir / "job.log",
    }


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes allowed root: {path}")
    return resolved


def uploaded_file_paths(spec: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in UPLOADED_FILE_KEYS:
        value = spec.get(key)
        if value:
            paths.append(ensure_within(Path(str(value)), REMOTE_CONTENT_ROOT))
    return paths


def delete_uploaded_files(uploaded_paths: list[Path], log_handle: Any | None = None) -> None:
    for uploaded_path in uploaded_paths:
        try:
            uploaded_path.unlink(missing_ok=True)
        except OSError as error:
            message = f"warning: unable to delete uploaded file {uploaded_path}: {error}"
            if log_handle is None:
                print(message, file=sys.stderr)
            else:
                log_handle.write(f"[{utc_now()}] {message}\n")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    safe_extract_into(archive, destination)


def safe_extract_into(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
            target = (destination / member.name).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.name}")
        bundle.extractall(destination, members=members, filter="data")


def load_env_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    env_path = Path(path)
    if not env_path.is_file():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not SAFE_ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid environment assignment at {env_path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[name] = value
    return result


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_names() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def update_status(path: Path, **updates: Any) -> dict[str, Any]:
    status = read_json(path) if path.exists() else {}
    status.update(updates)
    status["updated_at"] = utc_now()
    atomic_write_json(path, status)
    return status


def run_logged_command(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
    log_handle: Any,
    label: str,
) -> int:
    if not command.strip():
        log_handle.write(f"[{utc_now()}] {label}: skipped\n")
        log_handle.flush()
        return 0
    log_handle.write(f"[{utc_now()}] {label}: {command}\n")
    log_handle.flush()
    completed = subprocess.run(
        ["bash", "-lc", f"set -eo pipefail\n{command}"],
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_handle.write(f"[{utc_now()}] {label} exit code: {completed.returncode}\n")
    log_handle.flush()
    return int(completed.returncode)


def worker(spec_path: Path) -> int:
    spec = read_json(spec_path)
    job_id = validate_job_id(str(spec["job_id"]))
    paths = job_paths(job_id)
    workdir = ensure_within(paths["repo"] / str(spec.get("workdir", ".")), paths["repo"])
    if not workdir.is_dir():
        update_status(paths["status"], state="FAILED_SETUP", message=f"Missing workdir: {workdir}")
        return 2

    env = os.environ.copy()
    uploaded_paths = uploaded_file_paths(spec)
    env_file_value = spec.get("env_file_path")
    try:
        env_values = load_env_file(env_file_value)
    except (OSError, ValueError) as error:
        delete_uploaded_files(uploaded_paths)
        update_status(
            paths["status"],
            state="FAILED_SETUP",
            message=f"Unable to load uploaded environment: {error}",
            finished_at=utc_now(),
        )
        return 2
    env.update(env_values)
    env["PYTHONUNBUFFERED"] = "1"
    env["TIMELENS2_COLAB_JOB_ID"] = job_id
    env["TIMELENS2_COLAB_RUN_DIR"] = str(paths["run_dir"])
    env["TIMELENS2_SOURCE_REVISION"] = str(spec.get("source_revision") or "unknown")
    env["TIMELENS2_SOURCE_ARCHIVE_SHA256"] = str(spec.get("archive_sha256") or "unknown")
    if env_values:
        env["TIMELENS2_ENV_LOADED"] = "1"
    youtube_cookies_value = spec.get("youtube_cookies_path")
    if youtube_cookies_value:
        youtube_cookies_path = ensure_within(
            Path(str(youtube_cookies_value)), REMOTE_CONTENT_ROOT
        )
        if not youtube_cookies_path.is_file():
            delete_uploaded_files(uploaded_paths)
            update_status(
                paths["status"],
                state="FAILED_SETUP",
                message=f"Uploaded YouTube cookies not found: {youtube_cookies_path}",
                finished_at=utc_now(),
            )
            return 2
        os.chmod(youtube_cookies_path, 0o600)
        env["YTDLP_COOKIES_FILE"] = str(youtube_cookies_path)
    env.setdefault("VLM_VIDEO_DECODE_BACKEND", "pyav")
    detected_gpus = gpu_names()
    if any("T4" in name.upper() for name in detected_gpus):
        env.setdefault("TIMELENS2_T4_SAFE_MODE", "1")
    data_root = REMOTE_CONTENT_ROOT / "timelens2-data"
    data_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("TIMELENS2_DATA_ROOT", str(data_root))

    with paths["log"].open("a", encoding="utf-8", buffering=1) as log_handle:
        try:
            log_handle.write(f"[{utc_now()}] worker pid={os.getpid()} job={job_id}\n")
            log_handle.write(f"[{utc_now()}] python={sys.version.split()[0]} cwd={workdir}\n")
            log_handle.write(
                f"[{utc_now()}] gpu_names={detected_gpus!r} "
                f"video_backend={env['VLM_VIDEO_DECODE_BACKEND']} "
                f"t4_safe_mode={env.get('TIMELENS2_T4_SAFE_MODE', '0')}\n"
            )
            resume_checkpoint_value = spec.get("resume_checkpoint_path")
            if resume_checkpoint_value:
                resume_checkpoint = ensure_within(
                    Path(str(resume_checkpoint_value)), REMOTE_CONTENT_ROOT
                )
                if not resume_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Uploaded resume checkpoint not found: {resume_checkpoint}"
                    )
                expected_checkpoint_hash = str(spec.get("resume_checkpoint_sha256") or "")
                actual_checkpoint_hash = sha256_file(resume_checkpoint)
                if actual_checkpoint_hash != expected_checkpoint_hash:
                    raise ValueError(
                        "Resume checkpoint SHA-256 mismatch: "
                        f"expected {expected_checkpoint_hash}, got {actual_checkpoint_hash}"
                    )
                checkpoint_destination = ensure_within(
                    Path(
                        env.get(
                            "TIMELENS2_OMTG_OUTPUT_ROOT",
                            str(REMOTE_CONTENT_ROOT / "timelens2-experiment-outputs/omtg_search"),
                        )
                    ),
                    REMOTE_CONTENT_ROOT,
                )
                safe_extract_into(resume_checkpoint, checkpoint_destination)
                log_handle.write(
                    f"[{utc_now()}] restored checkpoint {actual_checkpoint_hash} "
                    f"into {checkpoint_destination}\n"
                )
            subprocess.run(
                ["nvidia-smi"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            update_status(paths["status"], state="RUNNING_SETUP", worker_pid=os.getpid())
            setup_code = run_logged_command(
                str(spec.get("setup_command", "")),
                cwd=workdir,
                env=env,
                log_handle=log_handle,
                label="setup",
            )
            if setup_code != 0:
                update_status(
                    paths["status"],
                    state="FAILED_SETUP",
                    exit_code=setup_code,
                    finished_at=utc_now(),
                )
                return setup_code

            update_status(paths["status"], state="DOWNLOADING_DATASET")
            dataset_code = run_logged_command(
                str(spec.get("dataset_command", "")),
                cwd=workdir,
                env=env,
                log_handle=log_handle,
                label="dataset",
            )
            if dataset_code != 0:
                update_status(
                    paths["status"],
                    state="FAILED_DATASET",
                    exit_code=dataset_code,
                    finished_at=utc_now(),
                )
                return dataset_code

            update_status(paths["status"], state="RUNNING")
            run_code = run_logged_command(
                str(spec["run_command"]),
                cwd=workdir,
                env=env,
                log_handle=log_handle,
                label="run",
            )
            final_state = "SUCCEEDED" if run_code == 0 else "FAILED"
            update_status(
                paths["status"],
                state=final_state,
                exit_code=run_code,
                finished_at=utc_now(),
            )
            return run_code
        except BaseException as error:
            traceback.print_exc(file=log_handle)
            update_status(
                paths["status"],
                state="FAILED",
                message=f"{type(error).__name__}: {error}",
                finished_at=utc_now(),
            )
            return 1
        finally:
            delete_uploaded_files(uploaded_paths, log_handle)


def launch(spec: dict[str, Any]) -> dict[str, Any]:
    job_id = validate_job_id(str(spec["job_id"]))
    paths = job_paths(job_id)
    archive = Path(str(spec["archive_path"]))
    expected_hash = str(spec["archive_sha256"])
    if not archive.is_file():
        raise FileNotFoundError(f"Uploaded archive not found: {archive}")
    actual_hash = sha256_file(archive)
    if actual_hash != expected_hash:
        raise ValueError(f"Archive SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
    if paths["run_dir"].exists():
        raise FileExistsError(f"Run directory already exists: {paths['run_dir']}")

    paths["run_dir"].mkdir(parents=True)
    safe_extract(archive, paths["repo"])
    atomic_write_json(paths["spec"], spec)
    initial = {
        "job_id": job_id,
        "state": "STARTING",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "archive_sha256": actual_hash,
        "source_revision": spec.get("source_revision"),
        "source_dirty": bool(spec.get("source_dirty", False)),
        "run_dir": str(paths["run_dir"]),
        "repo_dir": str(paths["repo"]),
        "log_path": str(paths["log"]),
    }
    atomic_write_json(paths["status"], initial)

    controller = paths["repo"] / "evaluation" / "scripts" / "colab_remote_job.py"
    if not controller.is_file():
        raise FileNotFoundError(f"Remote worker missing from source archive: {controller}")
    process = subprocess.Popen(
        [sys.executable, str(controller), "--worker", str(paths["spec"])],
        cwd=paths["run_dir"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return update_status(paths["status"], worker_pid=process.pid)


def cancel(spec: dict[str, Any]) -> dict[str, Any]:
    job_id = validate_job_id(str(spec["job_id"]))
    paths = job_paths(job_id)
    status = read_json(paths["status"])
    pid = int(status.get("worker_pid", 0) or 0)
    if status.get("state") in TERMINAL_STATES:
        return status
    if pid > 0 and process_alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(20):
            if not process_alive(pid):
                break
            time.sleep(0.25)
        if process_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    original_spec = read_json(paths["spec"])
    delete_uploaded_files(uploaded_file_paths(original_spec))
    return update_status(paths["status"], state="CANCELED", finished_at=utc_now())


def archive_outputs(spec: dict[str, Any]) -> dict[str, Any]:
    job_id = validate_job_id(str(spec["job_id"]))
    paths = job_paths(job_id)
    original_spec = read_json(paths["spec"])
    output_paths = original_spec.get("output_paths", ["evaluation/outputs"])
    if not isinstance(output_paths, list):
        raise ValueError("output_paths must be a list")
    archive = REMOTE_CONTENT_ROOT / f"timelens2_{job_id}_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for relative_value in output_paths:
            relative = Path(str(relative_value))
            if relative.is_absolute():
                raise ValueError(f"Output path must be relative to repository root: {relative}")
            source = ensure_within(paths["repo"] / relative, paths["repo"])
            if source.exists():
                bundle.add(source, arcname=str(relative))
        for key, name in (
            ("status", "_colab/status.json"),
            ("log", "_colab/job.log"),
            ("spec", "_colab/job_spec.json"),
        ):
            if paths[key].exists():
                bundle.add(paths[key], arcname=name)
    result = {
        "job_id": job_id,
        "archive_path": str(archive),
        "archive_sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
    }
    atomic_write_json(paths["run_dir"] / "output_archive.json", result)
    return result


def controller(spec_path: Path) -> int:
    spec = read_json(spec_path)
    action = str(spec.get("action", "launch"))
    if action == "launch":
        result = launch(spec)
    elif action == "cancel":
        result = cancel(spec)
    elif action == "archive":
        result = archive_outputs(spec)
    else:
        raise ValueError(f"Unknown controller action: {action}")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, help="Run a detached worker using this saved spec")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    # `colab exec -f` evaluates this file inside an IPython kernel. The kernel
    # leaves its own arguments (notably `-f kernel-*.json`) in sys.argv.
    # Ignore those while preserving our explicit worker/spec arguments.
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    if args.worker is not None:
        return worker(args.worker)
    return controller(args.spec)


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
