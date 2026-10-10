"""Build deterministic, allowlisted archives without local caches or environments."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "DATA_LICENSE.md",
    "DATA_CARD.md",
    "CHANGELOG.md",
    "VERIFICATION.md",
    "CITATION.cff",
    "requirements-lock.txt",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
}
ALLOWED_SUFFIXES = {
    "src": {".py", ".yaml"},
    "tests": {".py"},
    "scripts": {".py"},
    "configs": {".json"},
    "provenance": {".json", ".xml"},
    ".github": {".yml"},
}
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
    for name in sorted(ROOT_FILES):
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing release file: {name}")
        yield path
    for directory, suffixes in ALLOWED_SUFFIXES.items():
        for path in sorted((root / directory).rglob("*")):
            if (
                path.is_file()
                and path.suffix in suffixes
                and not any(
                    part in {"__pycache__", ".pytest_cache", ".ruff_cache", "build"}
                    or part.endswith(".egg-info")
                    for part in path.relative_to(root).parts
                )
            ):
                yield path
    for dataset in ("spider", "bird"):
        for name in ("annotations.jsonl", "seeds.jsonl", "schemas.json", "columns.json"):
            path = root / "data" / dataset / name
            if not path.is_file():
                raise ValueError(f"Missing benchmark file: {dataset}/{name}")
            yield path


def audit(root, include_values=False):
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


def write_archive(path, files, root):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path.name}")
    with zipfile.ZipFile(path, "x") as archive:
        for file in sorted(files):
            info = zipfile.ZipInfo(
                "clarity-public/" + file.relative_to(root).as_posix(), (2026, 10, 9, 0, 0, 0)
            )
            info.external_attr = 0o100644 << 16
            info.compress_type = (
                zipfile.ZIP_STORED if file.suffix == ".gz" else zipfile.ZIP_DEFLATED
            )
            archive.writestr(info, file.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError(f"Archive integrity check failed for {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-values", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root, args.include_values)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        main_archive = args.output / "clarity-spider-bird-0.1.0.zip"
        write_archive(main_archive, release_files(root), root)
        if args.include_values:
            values = [root / "data" / dataset / "values.json.gz" for dataset in ("spider", "bird")]
            write_archive(args.output / "clarity-values-0.1.0.zip", values, root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
