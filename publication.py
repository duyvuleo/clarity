"""Read a frozen publication inventory; refresh it only from the reviewed Git index."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

MANIFEST = "PUBLICATION.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 or ord(c) == 127 for c in name)
        or PurePosixPath(name).is_absolute()
        or any(p in {"", ".", ".."} for p in name.split("/"))
    ):
        raise ValueError("Invalid publication path")
    return name


def read_regular(root, name):
    name = canonical(name)
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink in publication input: {name}")
    if not path.is_file():
        raise ValueError(f"Missing publication input: {name}")
    return path.read_bytes()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate publication inventory key")
        result[key] = value
    return result


def snapshot(root, software_only=False):
    """Return verified bytes, including the inventory, without filesystem discovery."""
    root = Path(root).resolve()
    raw = read_regular(root, MANIFEST)
    inventory = json.loads(raw, object_pairs_hook=unique_pairs)
    if set(inventory) != {"schema_version", "files"} or inventory["schema_version"] != 1:
        raise ValueError("Unsupported publication inventory")
    if not isinstance(inventory["files"], dict) or not inventory["files"]:
        raise ValueError("Empty publication inventory")
    files = {MANIFEST: raw}
    for name, expected in inventory["files"].items():
        canonical(name)
        if (
            name == MANIFEST
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("Invalid publication inventory entry")
        if software_only and name.startswith("data/"):
            continue
        data = read_regular(root, name)
        if sha256(data) != expected:
            raise ValueError(f"Unreviewed change in publication input: {name}")
        files[name] = data
    return files


def stage(files, destination):
    for name, data in files.items():
        target = destination / canonical(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def from_index(root):
    """Explicit release-review action. Never called automatically by a build."""
    root = Path(root).resolve()
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root)
    if Path(top.decode().strip()).resolve() != root:
        raise ValueError("Publication inventory requires this repository's index")
    records = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    files = {}
    for record in records.split(b"\0"):
        if not record:
            continue
        header, raw_name = record.split(b"\t", 1)
        mode, oid, index_stage = header.split()
        name = canonical(raw_name.decode("utf-8"))
        if mode not in {b"100644", b"100755"} or index_stage != b"0":
            raise ValueError("Publication index contains a conflict or non-regular file")
        if name == MANIFEST:
            continue
        data = subprocess.check_output(["git", "cat-file", "blob", oid.decode()], cwd=root)
        if read_regular(root, name) != data:
            raise ValueError(f"Unstaged publication input: {name}")
        files[name] = sha256(data)
    if not files:
        raise ValueError("Empty publication index")
    (root / MANIFEST).write_text(
        json.dumps({"schema_version": 1, "files": dict(sorted(files.items()))}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-index", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    from_index(parser.parse_args().root)
