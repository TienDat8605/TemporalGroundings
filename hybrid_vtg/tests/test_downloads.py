import io
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

from hybrid_vtg.downloads import (
    SOURCES,
    _extract_tar,
    _extract_zip,
    _gdown_folder,
    _http_headers,
    _qvhighlights_video_paths,
    asset_paths,
    download_assets,
    resolve_targets,
)


def test_download_layout_is_one_predictable_assets_tree(tmp_path: Path):
    paths = asset_paths(tmp_path / "assets")
    assert paths["omtg"] == (tmp_path / "assets" / "datasets" / "omtg").resolve()
    assert paths["tacos"] == (tmp_path / "assets" / "datasets" / "tacos").resolve()
    assert paths["qvhighlights"] == (tmp_path / "assets" / "datasets" / "qvhighlights").resolve()
    assert paths["timelens2-4b"] == (tmp_path / "assets" / "checkpoints" / "timelens2-4b").resolve()
    assert paths["univtg"].name == "univtg-pretrained-clip-b32-4m"
    assert resolve_targets(()) == ("omtg", "tacos", "qvhighlights", "timelens2-4b", "univtg")


def test_tacos_uses_videomind_compressed_release():
    assert "yeliudev/VideoMind-Dataset" in SOURCES["tacos"]["annotation"]
    assert "videos_3fps_480_noaudio.tar.gz" in SOURCES["tacos"]["videos"]


def test_qvhighlights_selects_only_unique_test_video_paths(tmp_path: Path):
    annotation = tmp_path / "highlight_test_release.jsonl"
    annotation.write_text(
        '{"qid": 1, "vid": "Abc_60.0_210.0"}\n'
        '{"qid": 2, "vid": "Abc_60.0_210.0"}\n'
        '{"qid": 3, "vid": "-xyz_60.0_210.0"}\n',
        encoding="utf-8",
    )
    assert _qvhighlights_video_paths(annotation) == [
        "-/-xyz_60.0_210.0.mp4",
        "a/Abc_60.0_210.0.mp4",
    ]


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
