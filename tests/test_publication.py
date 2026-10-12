"""Regression checks for the publication boundary, independent of Git availability."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from publication import canonical, sha256, snapshot
from build_backend import normalize_sdist, verify_wheel


def inventory(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "PUBLICATION.json").write_text(
        json.dumps({"schema_version": 1, "files": {k: sha256(v) for k, v in files.items()}})
    )


def test_unreviewed_files_are_excluded_and_changes_fail(tmp_path):
    inventory(tmp_path, {"src/clarity/__init__.py": b"# reviewed\n"})
    (tmp_path / "src/clarity/unreviewed.py").write_text("# unreviewed\n")
    assert set(snapshot(tmp_path)) == {"PUBLICATION.json", "src/clarity/__init__.py"}
    (tmp_path / "src/clarity/__init__.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="Unreviewed change"):
        snapshot(tmp_path)


@pytest.mark.parametrize("name", ["../x", "/x", "x/../y", "x//y", "x\\y", "C:x", "x\n"])
def test_invalid_paths_are_rejected(name):
    with pytest.raises(ValueError, match="Invalid publication path"):
        canonical(name)


@pytest.mark.parametrize("parent", [False, True])
def test_symlinked_inputs_fail(tmp_path, parent):
    inventory(tmp_path, {"folder/file.py": b"# reviewed"})
    if parent:
        (tmp_path / "folder").rename(tmp_path / "elsewhere")
        (tmp_path / "folder").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    else:
        (tmp_path / "folder/file.py").rename(tmp_path / "real.py")
        (tmp_path / "folder/file.py").symlink_to(tmp_path / "real.py")
    with pytest.raises(ValueError, match="Symlink"):
        snapshot(tmp_path)


def test_duplicate_inventory_and_missing_software_fail(tmp_path):
    (tmp_path / "PUBLICATION.json").write_text('{"schema_version":1,"files":{},"files":{}}')
    with pytest.raises(ValueError, match="Duplicate"):
        snapshot(tmp_path)
    inventory(tmp_path, {"src/clarity/__init__.py": b"", "data/bird/seeds.jsonl": b"{}"})
    (tmp_path / "data/bird/seeds.jsonl").unlink()
    assert "src/clarity/__init__.py" in snapshot(tmp_path, software_only=True)
    (tmp_path / "src/clarity/__init__.py").unlink()
    with pytest.raises(ValueError, match="Missing"):
        snapshot(tmp_path, software_only=True)


def test_wheel_rejects_extra_module(tmp_path):
    path = tmp_path / "fixture.whl"
    files = {"src/clarity/__init__.py": b""}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("clarity/__init__.py", b"")
        archive.writestr("clarity/unreviewed.py", b"")
    with pytest.raises(ValueError, match="Unexpected wheel member"):
        verify_wheel(path, files)


def test_sdist_normalizes_headers_and_preserves_bytes(tmp_path):
    path = tmp_path / "fixture.tar.gz"
    data = b"# reviewed\n"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("fixture/file.py")
        entry.size = len(data)
        entry.uid = entry.gid = 1234
        entry.uname = entry.gname = "builder"
        entry.pax_headers = {"uname": "builder", "mtime": "123.45"}
        archive.addfile(entry, io.BytesIO(data))
    normalize_sdist(path, {"file.py": data})
    with tarfile.open(path) as archive:
        entry = archive.getmembers()[0]
        assert (entry.uid, entry.gid, entry.uname, entry.gname, entry.mtime) == (0, 0, "", "", 0)
        assert not entry.pax_headers
        assert archive.extractfile(entry).read() == data


def test_direct_zip_writer_rejects_arbitrary_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "release_packager", ROOT / "scripts/package_release.py"
    )
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    inventory(tmp_path, {"README.md": b"# reviewed"})
    extra = tmp_path / "unreviewed.py"
    extra.write_text("# unreviewed")
    with pytest.raises(ValueError, match="Archive members differ"):
        packager.write_archive(tmp_path / "out.zip", [extra], tmp_path)


def test_provenance_uses_unique_portable_identifiers():
    entries = json.loads((ROOT / "provenance/source-manifest.json").read_text())["files"]
    assert len(entries) == 49
    assert len({entry["source_id"] for entry in entries}) == len(entries)
    assert all("source" not in entry and entry["source_sha256"] for entry in entries)
