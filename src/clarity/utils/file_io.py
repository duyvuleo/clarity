"""UTF-8 JSON, JSONL and YAML I/O, including compressed value catalogs."""

import gzip
import json
from pathlib import Path

import yaml


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {Path(path).name}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {Path(path).name}:{line_number}")
            records.append(record)
    return records


def load_yaml(path, encoding="utf-8"):
    with Path(path).open(encoding=encoding) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("YAML configuration must contain an object")
    return data


def to_jsonl(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in data:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_jsonl_file(path):
    return Path(path).suffix == ".jsonl"


def is_json_file(path):
    return Path(path).suffix == ".json"
