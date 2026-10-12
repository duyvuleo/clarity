"""Build deterministic, allowlisted archives without local caches or environments."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tomllib
import zipfile
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from publication import read_regular, sha256, snapshot, stage

# Report only the file and rule name, never the matched material.
LOCAL_ROOT_PATTERN = "/" + "Users/" + "|/" + "home/"
RULES = {
    "private_infrastructure": re.compile(r"oraclecorp\.|ocid1\.|orahub\.oci"),
    "local_filesystem_path": re.compile(LOCAL_ROOT_PATTERN),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "token_literal": re.compile(r"\b(?:sk-|ghp_|hf_)[A-Za-z0-9_-]{24,}"),
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def release_files(root):
    for name in sorted(snapshot(root)):
        yield root / name


def publication_bytes(root, include_values=False):
    files = snapshot(root)
    if include_values:
        manifest = json.loads(files["provenance/data-manifest.json"])
        for info in manifest["datasets"].values():
            for entry in info["files"]:
                if entry.get("optional"):
                    name = entry["path"]
                    data = read_regular(root, name)
                    if sha256(data) != entry["sha256"]:
                        raise ValueError("Optional catalog hash mismatch")
                    files[name] = data
    return files


def audit(root, include_values=False):
    provenance = json.loads((root / "provenance/source-manifest.json").read_text())
    if any("source" in entry or "source_id" not in entry for entry in provenance["files"]):
        raise ValueError("Provenance must use portable source identifiers")
    manifest = json.loads((root / "provenance/data-manifest.json").read_text())
    files = list(release_files(root))
    if include_values:
        files.extend(root / "data" / dataset / "values.json.gz" for dataset in ("spider", "bird"))
    findings = []
    content_review = json.loads((root / "provenance/benchmark-content-review.json").read_text())
    reviewed_catalog = root / content_review["catalog"]
    reviewed_paths = (
        (
            content_review["all_matched_strings_verified_in_official_db"] is True
            and content_review["unverified_values"] == 0
            and digest(reviewed_catalog) == content_review["catalog_sha256"]
        )
        if include_values
        else False
    )
    for path in files:
        relative = str(path.relative_to(root))
        if path.is_symlink():
            findings.append({"file": relative, "rule": "symlink"})
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            previous = ""
            while chunk := handle.read(1024 * 1024):
                text = previous + chunk
                for name, rule in RULES.items():
                    if rule.search(text):
                        if (
                            name == "local_filesystem_path"
                            and path == reviewed_catalog
                            and reviewed_paths
                        ):
                            continue
                        findings.append({"file": relative, "rule": name})
                previous = chunk[-256:]
    for dataset, info in manifest["datasets"].items():
        allowed = set(info["database_ids"])
        directory = root / "data" / dataset
        for kind in ("schemas", "columns"):
            if set(json.loads((directory / (kind + ".json")).read_text())) != allowed:
                findings.append({"file": f"data/{dataset}/{kind}.json", "rule": "database_scope"})
        for entry in info["files"]:
            path = root / entry["path"]
            if not path.exists() and entry.get("optional"):
                continue
            if digest(path) != entry["sha256"]:
                findings.append({"file": entry["path"], "rule": "data_hash"})
        if include_values:
            with gzip.open(directory / "values.json.gz", "rt", encoding="utf-8") as handle:
                if set(json.load(handle)) != allowed:
                    findings.append(
                        {"file": f"data/{dataset}/values.json.gz", "rule": "database_scope"}
                    )
    if findings:
        raise ValueError(json.dumps(findings))
    return {
        "allowlisted_files": len(files),
        "content_rules": list(RULES),
        "findings": [],
        "values_checked": include_values,
        "verified_benchmark_content_exceptions": content_review["flagged_unique_values"]
        if include_values
        else 0,
    }


def _write_bytes(path, files):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path.name}")
    with zipfile.ZipFile(path, "x") as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo("clarity-public/" + name, (2026, 10, 12, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = (
                zipfile.ZIP_STORED if name.endswith(".gz") else zipfile.ZIP_DEFLATED
            )
            archive.writestr(info, data, compresslevel=9)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError(f"Archive integrity check failed for {path.name}")


def write_archive(path, files, root):
    """Validate direct callers as well as the CLI, then archive the audited bytes."""
    selected = {file.relative_to(root).as_posix() for file in files}
    values = {f"data/{dataset}/values.json.gz" for dataset in ("spider", "bird")}
    include_values = selected == values
    reviewed = publication_bytes(root, include_values)
    if selected != (values if include_values else set(reviewed)):
        raise ValueError("Archive members differ from the reviewed publication inventory")
    with tempfile.TemporaryDirectory(prefix="clarity-release-") as temporary:
        source = Path(temporary)
        stage(reviewed, source)
        audit(source, include_values)
        _write_bytes(path, {name: reviewed[name] for name in selected})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-values", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    reviewed = publication_bytes(root, args.include_values)
    with tempfile.TemporaryDirectory(prefix="clarity-release-") as temporary:
        source = Path(temporary)
        stage(reviewed, source)
        result = audit(source, args.include_values)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
        main_archive = args.output / f"clarity-spider-bird-{version}.zip"
        _write_bytes(
            main_archive,
            {name: data for name, data in reviewed.items() if not name.endswith("values.json.gz")},
        )
        if args.include_values:
            values = {
                name: data for name, data in reviewed.items() if name.endswith("values.json.gz")
            }
            _write_bytes(args.output / f"clarity-values-{version}.zip", values)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
