"""Official asset downloads with one predictable local layout."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import write_json

TARGETS = ("omtg", "tacos", "qvhighlights", "timelens2-4b", "univtg")

SOURCES = {
    "omtg": {
        "annotation": ("https://huggingface.co/datasets/insomnia7/omtg_bench/resolve/main/OMTGBench.tsv?download=true"),
        "videos": ("https://huggingface.co/datasets/insomnia7/omtg_bench/resolve/main/videos.zip?download=true"),
    },
    "qvhighlights": {
        "annotation": ("https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_test_release.jsonl"),
        "videos": "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz",
    },
    "tacos": {
        "annotation": ("https://drive.google.com/drive/folders/1aQ0mrXR7ZDfNiawqzQwgmzD3XNXUewDQ?usp=drive_link"),
        "videos": "https://datasets.d2.mpi-inf.mpg.de/MPII-Cooking-2/MPII-Cooking-2-videos.tar.gz",
        "terms": (
            "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/"
            "research/human-activity-recognition/mpii-cooking-2-dataset"
        ),
    },
    "timelens2-4b": {"repository": "MCG-NJU/TimeLens2-4B"},
    "univtg": {
        "checkpoint": ("https://drive.google.com/drive/folders/1-eGata6ZPV0A1BBsZpYyIooos9yjMx2f?usp=sharing"),
        "variant": "pretrained-clip-b32-4m",
    },
}


def asset_paths(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    return {
        "omtg": root / "datasets" / "omtg",
        "tacos": root / "datasets" / "tacos",
        "qvhighlights": root / "datasets" / "qvhighlights",
        "timelens2-4b": root / "checkpoints" / "timelens2-4b",
        "univtg": root / "checkpoints" / "univtg-pretrained-clip-b32-4m",
    }


def resolve_targets(values: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(values) or TARGETS
    unknown = sorted(set(selected) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown download targets: {', '.join(unknown)}")
    return tuple(value for value in TARGETS if value in selected)


def _http_headers(url: str, *, offset: int, hf_token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "hybrid-vtg/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    hostname = (urlparse(url).hostname or "").lower()
    if hf_token and (hostname == "huggingface.co" or hostname.endswith(".huggingface.co")):
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


def _download_http(url: str, destination: Path, *, hf_token: str | None = None) -> Path:
    from tqdm.auto import tqdm

    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers=_http_headers(url, offset=offset, hf_token=hf_token))
    print(f"download: {url}\n      -> {destination}")
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        initial = offset if append else 0
        content_length = response.headers.get("Content-Length")
        total = initial + int(content_length) if content_length else None
        with partial.open("ab" if append else "wb") as handle:
            with tqdm(
                total=total,
                initial=initial,
                desc=destination.name,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    progress.update(len(chunk))
    partial.replace(destination)
    return destination


def _safe_path(root: Path, member: str) -> Path:
    destination = (root / member).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"archive member escapes destination: {member}")
    return destination


def _extract_zip(archive: Path, destination: Path) -> None:
    from tqdm.auto import tqdm

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            _safe_path(destination, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"unsupported archive member: {member.filename}")
        for member in tqdm(members, desc=f"extract {archive.name}", unit="file"):
            handle.extract(member, destination)


def _extract_tar(archive: Path, destination: Path) -> None:
    from tqdm.auto import tqdm

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            _safe_path(destination, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported archive member: {member.name}")
        for member in tqdm(members, desc=f"extract {archive.name}", unit="file"):
            handle.extract(member, destination)


def _gdown_folder(url: str, destination: Path) -> list[Path]:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Google Drive downloads require: pip install -e '.[downloads]'") from error
    destination.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}\n      -> {destination}")
    result = gdown.download_folder(
        url=url,
        output=str(destination) + os.sep,
        quiet=False,
        remaining_ok=True,
    )
    files = [Path(value) for value in (result or ())]
    if not files and not any(path.is_file() for path in destination.rglob("*")):
        raise RuntimeError(f"Google Drive folder produced no files: {url}")
    return files


def _preflight_dependencies(selected: Sequence[str]) -> None:
    if selected and importlib.util.find_spec("tqdm") is None:
        raise RuntimeError("downloads require: pip install -e '.[downloads]'")
    if {"tacos", "univtg"}.intersection(selected) and importlib.util.find_spec("gdown") is None:
        raise RuntimeError("Google Drive downloads require: pip install -e '.[downloads]'")
    if "timelens2-4b" in selected and importlib.util.find_spec("huggingface_hub") is None:
        raise RuntimeError("TimeLens2 downloads require: pip install -e '.[downloads]'")


def _complete(destination: Path, source: dict[str, str], **extra: Any) -> dict[str, Any]:
    value = {"complete": True, "destination": str(destination), "source": source, **extra}
    write_json(destination / ".complete.json", value)
    return value


def _require_videos(destination: Path) -> None:
    suffixes = {".avi", ".mkv", ".mp4", ".webm"}
    if not any(path.is_file() and path.suffix.lower() in suffixes for path in destination.rglob("*")):
        raise FileNotFoundError(f"downloaded archive contains no supported videos under {destination}")


def _download_omtg(root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    _download_http(SOURCES["omtg"]["annotation"], destination / "OMTGBench.tsv", hf_token=hf_token)
    archive = _download_http(
        SOURCES["omtg"]["videos"],
        root / ".downloads" / "omtg-videos.zip",
        hf_token=hf_token,
    )
    _extract_zip(archive, destination / "videos")
    _require_videos(destination / "videos")
    value = _complete(destination, SOURCES["omtg"])
    archive.unlink()
    return value


def _download_qvhighlights(root: Path, destination: Path, _hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    annotation = destination / "annotations" / "highlight_test_release.jsonl"
    _download_http(SOURCES["qvhighlights"]["annotation"], annotation)
    archive = _download_http(
        SOURCES["qvhighlights"]["videos"],
        root / ".downloads" / "qvhighlights-videos.tar.gz",
    )
    _extract_tar(archive, destination / "videos")
    _require_videos(destination / "videos")
    value = _complete(destination, SOURCES["qvhighlights"])
    archive.unlink()
    return value


def _download_tacos(root: Path, destination: Path, _hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="hybrid-vtg-tacos-") as temporary:
        metadata = Path(temporary) / "metadata"
        _gdown_folder(SOURCES["tacos"]["annotation"], metadata)
        candidates = sorted(metadata.rglob("test.jsonl"))
        if not candidates:
            candidates = sorted(path for path in metadata.rglob("*.jsonl") if "test" in path.name.lower())
        if not candidates:
            raise FileNotFoundError("the official UniVTG TACoS metadata folder contains no test JSONL")
        annotation = destination / "annotations" / "test.jsonl"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], annotation)
    archive = _download_http(SOURCES["tacos"]["videos"], root / ".downloads" / "tacos-videos.tar.gz")
    _extract_tar(archive, destination / "videos")
    _require_videos(destination / "videos")
    value = _complete(destination, SOURCES["tacos"], terms_accepted=True)
    archive.unlink()
    return value


def _download_timelens2(_root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    print(f"download: https://huggingface.co/{SOURCES['timelens2-4b']['repository']}\n      -> {destination}")
    snapshot_download(
        repo_id=SOURCES["timelens2-4b"]["repository"],
        local_dir=destination,
        token=hf_token,
    )
    return _complete(destination, SOURCES["timelens2-4b"])


def _download_univtg(_root: Path, destination: Path, _hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    _gdown_folder(SOURCES["univtg"]["checkpoint"], destination)
    checkpoints = sorted(str(path.resolve()) for path in destination.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError("the official UniVTG folder contains no .ckpt file")
    return _complete(destination, SOURCES["univtg"], checkpoints=checkpoints)


DOWNLOADERS: dict[str, Callable[[Path, Path, str | None], dict[str, Any]]] = {
    "omtg": _download_omtg,
    "tacos": _download_tacos,
    "qvhighlights": _download_qvhighlights,
    "timelens2-4b": _download_timelens2,
    "univtg": _download_univtg,
}


def _hugging_face_token(*, login_requested: bool) -> str | None:
    installed = importlib.util.find_spec("huggingface_hub") is not None
    if not installed:
        if login_requested:
            raise RuntimeError("Hugging Face login requires: pip install -e '.[downloads]'")
        return os.environ.get("HF_TOKEN")

    from huggingface_hub import get_token, login

    if login_requested:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        else:
            login()
    return get_token()


def download_assets(
    root: Path,
    targets: Sequence[str],
    *,
    accept_licenses: bool,
    hf_login: bool = False,
) -> dict[str, Any]:
    selected = resolve_targets(targets)
    if not accept_licenses:
        raise ValueError("pass --accept-licenses after reviewing the source terms listed in README.md")
    _preflight_dependencies(selected)
    from tqdm.auto import tqdm

    hf_token = _hugging_face_token(login_requested=hf_login)
    root = root.expanduser().resolve()
    paths = asset_paths(root)
    completed = {}
    for target in tqdm(selected, desc="assets", unit="asset"):
        downloader = DOWNLOADERS[target]
        completed[target] = downloader(root, paths[target], hf_token)
    summary = {
        "root": str(root),
        "targets": list(selected),
        "paths": {target: str(paths[target]) for target in selected},
        "completed": completed,
    }
    write_json(root / "manifest.json", summary)
    return summary
