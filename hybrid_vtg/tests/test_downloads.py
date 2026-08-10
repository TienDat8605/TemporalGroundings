import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from hybrid_vtg.downloads import _extract_tar, _extract_zip, asset_paths, download_assets, resolve_targets


def test_download_layout_is_one_predictable_assets_tree(tmp_path: Path):
    paths = asset_paths(tmp_path / "assets")
    assert paths["omtg"] == (tmp_path / "assets" / "datasets" / "omtg").resolve()
    assert paths["tacos"] == (tmp_path / "assets" / "datasets" / "tacos").resolve()
    assert paths["qvhighlights"] == (tmp_path / "assets" / "datasets" / "qvhighlights").resolve()
    assert paths["timelens2-4b"] == (tmp_path / "assets" / "checkpoints" / "timelens2-4b").resolve()
    assert paths["univtg"].name == "univtg-pretrained-clip-b32-4m"
    assert resolve_targets(()) == ("omtg", "tacos", "qvhighlights", "timelens2-4b", "univtg")


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
