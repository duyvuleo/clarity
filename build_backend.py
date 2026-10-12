"""PEP 517 builds from verified software bytes, with portable sdist metadata."""

from contextlib import contextmanager
import gzip
import io
import os
from pathlib import Path
import tarfile
import tempfile
import zipfile

from setuptools import build_meta

from publication import canonical, snapshot, stage


@contextmanager
def reviewed_source():
    root = Path.cwd()
    files = snapshot(root, software_only=True)
    with tempfile.TemporaryDirectory(prefix="clarity-build-") as temporary:
        source = Path(temporary)
        stage(files, source)
        try:
            os.chdir(source)
            yield files
        finally:
            os.chdir(root)


def verify_wheel(path, files):
    expected = {name[4:]: data for name, data in files.items() if name.startswith("src/")}
    metadata = {"METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD"}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip():
            raise ValueError("Invalid wheel members")
        for name in names:
            canonical(name)
            if name in expected:
                if archive.read(name) != expected[name]:
                    raise ValueError("Wheel source differs from reviewed bytes")
            else:
                head, separator, tail = name.partition("/")
                if (
                    not separator
                    or not head.endswith(".dist-info")
                    or tail not in (metadata | {"licenses/LICENSE", "licenses/NOTICE.md"})
                ):
                    raise ValueError("Unexpected wheel member")
        if not set(expected).issubset(names):
            raise ValueError("Wheel is missing reviewed package files")


def normalize_sdist(path, files):
    generated = {"PKG-INFO", "setup.cfg"}
    egg_info = {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "top_level.txt",
    }
    seen = set()
    with tarfile.open(path, "r:gz") as source:
        members = []
        roots = {member.name.split("/")[0] for member in source.getmembers()}
        if len(roots) != 1:
            raise ValueError("Invalid sdist root")
        for member in source.getmembers():
            canonical(member.name)
            if member.name in seen or not (member.isfile() or member.isdir()):
                raise ValueError("Invalid sdist member")
            seen.add(member.name)
            relative = member.name.partition("/")[2]
            data = source.extractfile(member).read() if member.isfile() else None
            if data is not None:
                if relative in files:
                    if data != files[relative]:
                        raise ValueError("Sdist source differs from reviewed bytes")
                elif relative not in generated and not (
                    relative.startswith("src/clarity_nl2sql.egg-info/")
                    and relative.rsplit("/", 1)[-1] in egg_info
                    and relative.count("/") == 2
                ):
                    raise ValueError("Unexpected sdist member")
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mtime = 0
            member.pax_headers = {}
            member.mode = 0o644 if member.isfile() else 0o755
            members.append((member, data))
    root = next(iter(roots))
    if not {root + "/" + name for name in files}.issubset(seen):
        raise ValueError("Sdist is missing reviewed software files")
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as target:
            for member, data in members:
                target.addfile(member, io.BytesIO(data) if data is not None else None)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    output = str(Path(wheel_directory).resolve())
    metadata = str(Path(metadata_directory).resolve()) if metadata_directory else None
    with reviewed_source() as files:
        name = build_meta.build_wheel(output, config_settings, metadata)
        verify_wheel(Path(output) / name, files)
    return name


def build_sdist(sdist_directory, config_settings=None):
    output = str(Path(sdist_directory).resolve())
    with reviewed_source() as files:
        name = build_meta.build_sdist(output, config_settings)
        normalize_sdist(Path(output) / name, files)
    return name


def get_requires_for_build_wheel(config_settings=None):
    with reviewed_source():
        return build_meta.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    with reviewed_source():
        return build_meta.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    output = str(Path(metadata_directory).resolve())
    with reviewed_source():
        return build_meta.prepare_metadata_for_build_wheel(output, config_settings)


# Editable installs intentionally retain the developer's persistent source checkout.
build_editable = build_meta.build_editable
get_requires_for_build_editable = build_meta.get_requires_for_build_editable
prepare_metadata_for_build_editable = build_meta.prepare_metadata_for_build_editable
