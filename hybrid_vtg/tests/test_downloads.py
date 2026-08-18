import io
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

from hybrid_vtg.downloads import (
    SOURCES,
    _download_http,
    _extract_tar,
    _extract_zip,
    _gdown_folder,
    _http_headers,
    _qvhighlights_video_ids,
    _retain_tacos_test_videos,
    _source_matches,
    _tacos_test_video_ids,
    _verify_qvhighlights_videos,
    asset_paths,
    download_assets,
    resolve_targets,
)


def test_download_layout_is_one_predictable_assets_tree(tmp_path: Path):
    paths = asset_paths(tmp_path / "assets")
    assert paths["omtg"] == (tmp_path / "assets" / "datasets" / "omtg").resolve()
    assert paths["tacos"] == (tmp_path / "assets" / "datasets" / "tacos").resolve()
    assert paths["qvhighlights"] == (tmp_path / "assets" / "datasets" / "qvhighlights").resolve()
    assert paths["qvhighlights-timelens"] == (tmp_path / "assets" / "datasets" / "qvhighlights-timelens").resolve()
    assert paths["momentseeker"] == (tmp_path / "assets" / "datasets" / "momentseeker").resolve()
    assert paths["unitime"] == (tmp_path / "assets" / "checkpoints" / "unitime").resolve()
    assert paths["timelens2-4b"] == (tmp_path / "assets" / "checkpoints" / "timelens2-4b").resolve()
    assert paths["timelens-8b"] == (tmp_path / "assets" / "checkpoints" / "timelens-8b").resolve()
    assert paths["timelens-7b"] == (tmp_path / "assets" / "checkpoints" / "timelens-7b").resolve()
    assert paths["univtg"].name == "univtg-pretrained-clip-b32-4m"
    assert resolve_targets(()) == (
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


def test_tacos_uses_videomind_compressed_release():
    assert "yeliudev/VideoMind-Dataset" in SOURCES["tacos"]["annotation"]
    assert "videos_3fps_480_noaudio.tar.gz" in SOURCES["tacos"]["videos"]


def test_tacos_retains_only_test_videos_and_handles_camera_postfix(tmp_path: Path):
    annotation = tmp_path / "test.jsonl"
    annotation.write_text('{"vid": "s30-d52"}\n{"vid": "s31-d21"}\n', encoding="utf-8")
    videos = tmp_path / "videos"
    videos.mkdir()
    test_camera = videos / "s30-d52-cam-2.mp4"
    test_plain = videos / "s31-d21.mp4"
    train = videos / "s13-d21-cam-002.mp4"
    for path in (test_camera, test_plain, train):
        path.touch()

    test_ids = _tacos_test_video_ids(annotation)
    assert _retain_tacos_test_videos(videos, test_ids) == 2
    assert test_camera.is_file()
    assert test_plain.is_file()
    assert not train.exists()


def test_qvhighlights_selects_only_unique_test_video_ids(tmp_path: Path):
    annotation = tmp_path / "highlight_test_release.jsonl"
    annotation.write_text(
        '{"qid": 1, "vid": "Abc_60.0_210.0"}\n'
        '{"qid": 2, "vid": "Abc_60.0_210.0"}\n'
        '{"qid": 3, "vid": "-xyz_60.0_210.0"}\n',
        encoding="utf-8",
    )
    assert _qvhighlights_video_ids(annotation) == frozenset({"-xyz_60.0_210.0", "Abc_60.0_210.0"})


def test_qvhighlights_archive_must_contain_only_test_videos(tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "test-video.mp4").touch()
    assert _verify_qvhighlights_videos(videos, frozenset({"test-video"})) == 1
    (videos / "train-video.mp4").touch()
    with pytest.raises(ValueError, match="non-test videos"):
        _verify_qvhighlights_videos(videos, frozenset({"test-video"}))


def test_download_extractors_reject_parent_traversal(tmp_path: Path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="escapes destination"):
        _extract_zip(zip_path, tmp_path / "zip-output")

    tar_path = tmp_path / "bad.tar"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="escapes destination"):
        _extract_tar(tar_path, tmp_path / "tar-output")


def test_download_requires_explicit_license_acceptance_before_network_access(tmp_path: Path):
    with pytest.raises(ValueError, match="accept-licenses"):
        download_assets(tmp_path / "assets", ("omtg",), accept_licenses=False)


def test_hugging_face_token_is_only_sent_to_hugging_face():
    headers = _http_headers("https://huggingface.co/org/repo/file", offset=12, hf_token="secret")
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Range"] == "bytes=12-"

    external = _http_headers("https://example.com/file", offset=0, hf_token="secret")
    assert "Authorization" not in external
    assert "Range" not in external


def test_download_marker_sources_survive_json_tuple_conversion():
    source = {"video_shards": ("first", "second")}
    serialized = {"video_shards": ["first", "second"]}
    assert _source_matches(serialized, source)


def test_aria2_credentials_are_not_exposed_in_process_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    destination = tmp_path / "archive.tar.gz"
    captured = {}

    monkeypatch.setattr("hybrid_vtg.downloads.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, check):
        assert check is True
        captured["command"] = command
        input_path = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--input-file=")))
        captured["input"] = input_path.read_text(encoding="utf-8")
        captured["mode"] = input_path.stat().st_mode & 0o777
        destination.touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    _download_http("https://huggingface.co/org/repo/file", destination, hf_token="secret")

    assert "secret" not in " ".join(captured["command"])
    assert "header=Authorization: Bearer secret" in captured["input"]
    assert captured["mode"] == 0o600
    assert not any(tmp_path.glob(".aria2-input-*"))


def test_gdown_folder_supports_legacy_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    received = {}

    def legacy_download_folder(url, output, quiet):
        received.update(url=url, output=output, quiet=quiet)
        result = Path(output) / "test.jsonl"
        result.write_text("{}\n", encoding="utf-8")
        return [str(result)]

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download_folder=legacy_download_folder))
    files = _gdown_folder("https://drive.google.com/folder", tmp_path / "metadata")

    assert files == [tmp_path / "metadata" / "test.jsonl"]
    assert received["quiet"] is False
