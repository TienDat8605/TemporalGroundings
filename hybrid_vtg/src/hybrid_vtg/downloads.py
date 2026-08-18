"""Official asset downloads with one predictable local layout."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
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

TARGETS = (
    "omtg",
    "tacos",
    "qvhighlights",
    "qvhighlights-timelens",
    "momentseeker",
    "unitime",
    "timelens2-4b",
    "timelens-8b",
    "timelens-7b",
    "univtg",
)
VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mp4", ".webm"})
_TACOS_VIDEO_NAME = re.compile(r"^(s\d+-d\d+)(?:-cam-\d+)?$", re.I)

SOURCES = {
    "omtg": {
        "annotation": ("https://huggingface.co/datasets/insomnia7/omtg_bench/resolve/main/OMTGBench.tsv?download=true"),
        "videos": ("https://huggingface.co/datasets/insomnia7/omtg_bench/resolve/main/videos.zip?download=true"),
    },
    "qvhighlights": {
        "annotation": ("https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_test_release.jsonl"),
        "videos": (
            "https://huggingface.co/datasets/jwnt4/qvhighlights-test/resolve/main/videos.tar.gz?download=true"
        ),
        "selection": "single archive containing the highlight_test_release.jsonl videos only",
    },
    "qvhighlights-timelens": {
        "annotation": (
            "https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/qvhighlights-timelens.json?download=true"
        ),
        "video_shards": (
            "https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/video_shards/qvhighlights/qvhighlights_shard_01.tar.gz?download=true",
            "https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/video_shards/qvhighlights/qvhighlights_shard_02.tar.gz?download=true",
            "https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/video_shards/qvhighlights/qvhighlights_shard_03.tar.gz?download=true",
        ),
        "variant": "TimeLens-Bench QVHighlights-TimeLens (1,511 videos, 1,541 queries)",
    },
    "momentseeker": {
        "annotation": (
            "https://huggingface.co/datasets/avery00/MomentSeeker/resolve/main/t2v.json?download=true"
        ),
        "video_shards": (
            "https://huggingface.co/datasets/avery00/MomentSeeker/resolve/main/videos.tar.gz.part_aa?download=true",
            "https://huggingface.co/datasets/avery00/MomentSeeker/resolve/main/videos.tar.gz.part_ab?download=true",
            "https://huggingface.co/datasets/avery00/MomentSeeker/resolve/main/videos.tar.gz.part_ac?download=true",
        ),
        "variant": "MomentSeeker Text-to-Moment (265 long videos, 1,000 queries)",
    },
    "tacos": {
        "annotation": (
            "https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/resolve/main/tacos/test.jsonl?download=true"
        ),
        "videos": (
            "https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/resolve/main/"
            "tacos/videos_3fps_480_noaudio.tar.gz?download=true"
        ),
        "variant": "VideoMind 3 FPS, 480p, no audio",
        "upstream": (
            "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/"
            "research/vision-and-language/tacos-multi-level-corpus"
        ),
    },
    "unitime": {
        "adapter_repository": "zeqianli/UniTime",
        "base_repository": "Qwen/Qwen2-VL-7B-Instruct",
    },
    "timelens2-4b": {"repository": "MCG-NJU/TimeLens2-4B"},
    "timelens-8b": {"repository": "TencentARC/TimeLens-8B"},
    "timelens-7b": {"repository": "TencentARC/TimeLens-7B"},
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
        "qvhighlights-timelens": root / "datasets" / "qvhighlights-timelens",
        "momentseeker": root / "datasets" / "momentseeker",
        "unitime": root / "checkpoints" / "unitime",
        "timelens2-4b": root / "checkpoints" / "timelens2-4b",
        "timelens-8b": root / "checkpoints" / "timelens-8b",
        "timelens-7b": root / "checkpoints" / "timelens-7b",
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
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}\n      -> {destination}")
    aria2c = shutil.which("aria2c")
    if aria2c:
        import subprocess

        cmd = [
            aria2c,
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "-d",
            str(destination.parent),
            "-o",
            destination.name,
            url,
        ]
        if hf_token:
            cmd.append(f"--header=Authorization: Bearer {hf_token}")
        subprocess.run(cmd, check=True)
        return destination

    from tqdm.auto import tqdm

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers=_http_headers(url, offset=offset, hf_token=hf_token))
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


def _extract_tar(
    archive: Path,
    destination: Path,
    *,
    include: Callable[[str], bool] | None = None,
) -> None:
    from tqdm.auto import tqdm

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            _safe_path(destination, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported archive member: {member.name}")
        selected = (
            members
            if include is None
            else [member for member in members if member.isfile() and include(member.name)]
        )
        for member in tqdm(selected, desc=f"extract {archive.name}", unit="file"):
            handle.extract(member, destination)


def _gdown_folder(url: str, destination: Path) -> list[Path]:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Google Drive downloads require: pip install -e '.[downloads]'") from error
    destination.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}\n      -> {destination}")
    options: dict[str, Any] = {
        "url": url,
        "output": str(destination) + os.sep,
        "quiet": False,
    }
    if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
        options["remaining_ok"] = True
    result = gdown.download_folder(**options)
    files = [Path(value) for value in (result or ())]
    if not files and not any(path.is_file() for path in destination.rglob("*")):
        raise RuntimeError(f"Google Drive folder produced no files: {url}")
    return files


def _preflight_dependencies(selected: Sequence[str]) -> None:
    if selected and importlib.util.find_spec("tqdm") is None:
        raise RuntimeError("downloads require: pip install -e '.[downloads]'")
    if "univtg" in selected and importlib.util.find_spec("gdown") is None:
        raise RuntimeError("Google Drive downloads require: pip install -e '.[downloads]'")
    hf_targets = {"unitime", "timelens2-4b", "timelens-8b", "timelens-7b"}
    if hf_targets.intersection(selected) and importlib.util.find_spec("huggingface_hub") is None:
        raise RuntimeError("Hugging Face checkpoint downloads require: pip install -e '.[downloads]'")


def _complete(destination: Path, source: dict[str, str], **extra: Any) -> dict[str, Any]:
    value = {"complete": True, "destination": str(destination), "source": source, **extra}
    write_json(destination / ".complete.json", value)
    return value


def _require_videos(destination: Path) -> None:
    if not any(path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES for path in destination.rglob("*")):
        raise FileNotFoundError(f"downloaded archive contains no supported videos under {destination}")


def _tacos_test_video_ids(annotation: Path) -> frozenset[str]:
    video_ids = set()
    with annotation.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            video_id = str(record.get("vid", "")).strip().lower()
            if not _TACOS_VIDEO_NAME.fullmatch(video_id):
                raise ValueError(f"invalid TACoS video ID on line {line_number}: {video_id!r}")
            video_ids.add(video_id)
    if not video_ids:
        raise ValueError(f"TACoS test annotation is empty: {annotation}")
    return frozenset(video_ids)


def _tacos_video_id(path: str | Path) -> str | None:
    match = _TACOS_VIDEO_NAME.fullmatch(Path(path).stem)
    return match.group(1).lower() if match else None


def _retain_tacos_test_videos(videos: Path, test_ids: frozenset[str]) -> int:
    found = set()
    for path in videos.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        video_id = _tacos_video_id(path)
        if video_id in test_ids:
            found.add(video_id)
        else:
            path.unlink()
    missing = sorted(test_ids - found)
    if missing:
        raise FileNotFoundError(f"compressed TACoS archive is missing {len(missing)} test videos: {missing[:5]}")
    for directory in sorted((path for path in videos.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return len(found)


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


def _qvhighlights_video_ids(annotation: Path) -> frozenset[str]:
    video_ids = set()
    with annotation.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            video_id = str(record.get("vid", "")).strip()
            if not video_id or "/" in video_id or "\\" in video_id:
                raise ValueError(f"invalid QVHighlights video ID on line {line_number}: {video_id!r}")
            video_ids.add(video_id)
    if not video_ids:
        raise ValueError(f"QVHighlights test annotation is empty: {annotation}")
    return frozenset(video_ids)


def _verify_qvhighlights_videos(videos: Path, expected: frozenset[str]) -> int:
    downloaded = {
        path.stem
        for path in videos.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }
    missing = sorted(expected - downloaded)
    if missing:
        raise FileNotFoundError(f"QVHighlights test archive is missing {len(missing)} videos: {missing[:5]}")
    unexpected = sorted(downloaded - expected)
    if unexpected:
        raise ValueError(f"QVHighlights test archive contains {len(unexpected)} non-test videos: {unexpected[:5]}")
    return len(downloaded)


def _download_qvhighlights(root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("source") == SOURCES["qvhighlights"]:
            return value
    annotation = destination / "annotations" / "highlight_test_release.jsonl"
    _download_http(SOURCES["qvhighlights"]["annotation"], annotation)
    video_ids = _qvhighlights_video_ids(annotation)
    archive = _download_http(
        SOURCES["qvhighlights"]["videos"],
        root / ".downloads" / "qvhighlights-test-videos.tar.gz",
        hf_token=hf_token,
    )
    videos = destination / "videos"
    destination.mkdir(parents=True, exist_ok=True)
    if videos.exists():
        shutil.rmtree(videos)
    with tempfile.TemporaryDirectory(prefix=".qv-test-videos-", dir=destination) as temporary:
        staged = Path(temporary)
        _extract_tar(archive, staged)
        _require_videos(staged)
        video_count = _verify_qvhighlights_videos(staged, video_ids)
        shutil.move(staged, videos)
    value = _complete(destination, SOURCES["qvhighlights"], video_count=video_count, split="test")
    archive.unlink()
    return value


def _download_tacos(root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    annotation = destination / "annotations" / "test.jsonl"
    videos = destination / "videos"
    if marker.is_file() and annotation.is_file() and videos.is_dir():
        test_ids = _tacos_test_video_ids(annotation)
        video_count = _retain_tacos_test_videos(videos, test_ids)
        return _complete(destination, SOURCES["tacos"], terms_accepted=True, split="test", video_count=video_count)

    _download_http(SOURCES["tacos"]["annotation"], annotation, hf_token=hf_token)
    test_ids = _tacos_test_video_ids(annotation)
    archive = _download_http(
        SOURCES["tacos"]["videos"],
        root / ".downloads" / "tacos-videos-3fps-480p-noaudio.tar.gz",
        hf_token=hf_token,
    )
    _extract_tar(archive, videos, include=lambda name: _tacos_video_id(name) in test_ids)
    _require_videos(videos)
    video_count = _retain_tacos_test_videos(videos, test_ids)
    value = _complete(
        destination,
        SOURCES["tacos"],
        terms_accepted=True,
        split="test",
        video_count=video_count,
    )
    archive.unlink()
    return value


def _download_qvhighlights_timelens(root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    annotation = destination / "qvhighlights-timelens.json"
    videos = destination / "videos"
    if marker.is_file() and annotation.is_file() and videos.is_dir():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("source") == SOURCES["qvhighlights-timelens"]:
            return value

    destination.mkdir(parents=True, exist_ok=True)
    _download_http(SOURCES["qvhighlights-timelens"]["annotation"], annotation, hf_token=hf_token)
    with annotation.open(encoding="utf-8") as handle:
        data = json.load(handle)
    expected_video_ids = frozenset(data.keys())

    videos.mkdir(parents=True, exist_ok=True)
    for shard_idx, shard_url in enumerate(SOURCES["qvhighlights-timelens"]["video_shards"], start=1):
        archive_name = f"qvhighlights-timelens-shard-{shard_idx:02d}.tar.gz"
        archive = _download_http(shard_url, root / ".downloads" / archive_name, hf_token=hf_token)
        _extract_tar(archive, videos)
        archive.unlink()

    _require_videos(videos)
    found_count = len({p.stem for p in videos.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES})
    value = _complete(
        destination,
        SOURCES["qvhighlights-timelens"],
        video_count=found_count,
        expected_videos=len(expected_video_ids),
    )
    return value


def _download_momentseeker(root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    import subprocess

    marker = destination / ".complete.json"
    annotation = destination / "t2v.json"
    videos = destination / "videos"
    if marker.is_file() and annotation.is_file() and videos.is_dir():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("source") == SOURCES["momentseeker"]:
            return value

    destination.mkdir(parents=True, exist_ok=True)
    _download_http(SOURCES["momentseeker"]["annotation"], annotation, hf_token=hf_token)
    with annotation.open(encoding="utf-8") as handle:
        data = json.load(handle)
    expected_video_ids = frozenset(Path(item.get("src_video_path", "")).stem for item in data)

    downloads_dir = root / ".downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for shard_idx, shard_url in enumerate(SOURCES["momentseeker"]["video_shards"], start=1):
        part_name = f"momentseeker-part-{shard_idx:02d}"
        part_path = _download_http(shard_url, downloads_dir / part_name, hf_token=hf_token)
        parts.append(part_path)

    combined_tar = downloads_dir / "momentseeker-videos.tar.gz"
    concat_cmd = f"cat {' '.join(str(p) for p in parts)} > {combined_tar}"
    subprocess.run(concat_cmd, shell=True, check=True)
    for p in parts:
        p.unlink(missing_ok=True)

    _extract_tar(combined_tar, destination)
    combined_tar.unlink(missing_ok=True)
    _require_videos(videos)

    found_count = len({p.stem for p in videos.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES})
    value = _complete(
        destination,
        SOURCES["momentseeker"],
        video_count=found_count,
        expected_videos=len(expected_video_ids),
        split="test",
    )
    return value


def _snapshot(repository: str, destination: Path, hf_token: str | None) -> None:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    print(f"download: https://huggingface.co/{repository}\n      -> {destination}")
    snapshot_download(repo_id=repository, local_dir=destination, token=hf_token)


def _download_unitime(_root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    adapter = destination / "adapter"
    base = destination / "qwen2-vl-7b"
    _snapshot(SOURCES["unitime"]["adapter_repository"], adapter, hf_token)
    _snapshot(SOURCES["unitime"]["base_repository"], base, hf_token)
    return _complete(destination, SOURCES["unitime"], adapter=str(adapter), base=str(base))


def _download_timelens2(_root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    _snapshot(SOURCES["timelens2-4b"]["repository"], destination, hf_token)
    return _complete(destination, SOURCES["timelens2-4b"])


def _download_timelens7(_root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    _snapshot(SOURCES["timelens-7b"]["repository"], destination, hf_token)
    return _complete(destination, SOURCES["timelens-7b"])


def _download_timelens8(_root: Path, destination: Path, hf_token: str | None) -> dict[str, Any]:
    marker = destination / ".complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    _snapshot(SOURCES["timelens-8b"]["repository"], destination, hf_token)
    return _complete(destination, SOURCES["timelens-8b"])


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
    "qvhighlights-timelens": _download_qvhighlights_timelens,
    "momentseeker": _download_momentseeker,
    "unitime": _download_unitime,
    "timelens2-4b": _download_timelens2,
    "timelens-8b": _download_timelens8,
    "timelens-7b": _download_timelens7,
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
