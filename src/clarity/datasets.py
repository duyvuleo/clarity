"""Read and validate the two released benchmark subsets without model access."""

from collections import Counter
from pathlib import Path

from clarity.utils.file_io import load_json, load_jsonl

DATASETS = ("spider", "bird")


def dataset_directory(data_root, dataset):
    if dataset not in DATASETS:
        raise ValueError("Only spider and bird are supported")
    return Path(data_root) / dataset


def load_annotations(data_root, dataset):
    return load_jsonl(dataset_directory(data_root, dataset) / "annotations.jsonl")


def validate_dataset(data_root, dataset):
    directory = dataset_directory(data_root, dataset)
    schemas = load_json(directory / "schemas.json")
    columns = load_json(directory / "columns.json")
    annotations = load_annotations(data_root, dataset)
    seeds = load_jsonl(directory / "seeds.jsonl")
    if set(schemas) != set(columns):
        raise ValueError(f"{dataset}: schema and column database IDs differ")
    for kind, records in [("annotations", annotations), ("seeds", seeds)]:
        for index, record in enumerate(records, 1):
            db_id = record.get("db_id")
            if db_id not in schemas:
                raise ValueError(f"{dataset}/{kind}:{index}: unknown database ID")
            if record.get("dataset") is not None:
                raise ValueError(f"{dataset}/{kind}:{index}: unexpected external dataset")
            if not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"{dataset}/{kind}:{index}: missing question")
            if not isinstance(record.get("expectedQuery"), str):
                raise ValueError(f"{dataset}/{kind}:{index}: missing parent SQL")
            if kind == "annotations":
                conversation = record.get("conversation")
                if not isinstance(conversation, list) or not conversation:
                    raise ValueError(f"{dataset}/{kind}:{index}: missing conversation")
                for turn in conversation:
                    if (
                        not isinstance(turn, dict)
                        or turn.get("role") not in {"USER", "AGENT"}
                        or not isinstance(turn.get("content"), str)
                    ):
                        raise ValueError(f"{dataset}/{kind}:{index}: invalid conversation turn")
    return {
        "dataset": dataset,
        "annotations": len(annotations),
        "seeds": len(seeds),
        "databases": len(schemas),
        "usecases": dict(sorted(Counter(record["usecase"] for record in annotations).items())),
    }
