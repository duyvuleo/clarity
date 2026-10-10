"""Small command line interface for dataset checks and live generation."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import sys

from clarity.datasets import DATASETS, validate_dataset

SCENARIOS = (
    "concise helpful",
    "verbose helpful",
    "partially helpful",
    "not helpful - no column",
    "not helpful - random column",
)


def generate(config, output, seed=10, batch=False):
    from clarity.generation.task_flows.flows.conv_gen_eval_v1 import ConvUtteranceGenEvalsV1
    from clarity.utils.file_io import load_jsonl

    random.seed(seed)
    flow = ConvUtteranceGenEvalsV1(task=str(config), scenario="concise helpful")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = load_jsonl(flow.data_path)
    scenarios = SCENARIOS[:1] if "confusion" in flow.ac_type else SCENARIOS
    targets = [output / "raw.jsonl", output / "receipt.json"] + [
        output / (scenario.replace(" ", "_") + ".jsonl") for scenario in scenarios
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("Output files already exist; use a new output directory")
    errors, processed, accepted = 0, 0, 0
    with ExitStack() as stack:
        raw = stack.enter_context((output / "raw.jsonl").open("x", encoding="utf-8"))
        handles = [
            stack.enter_context(
                (output / (scenario.replace(" ", "_") + ".jsonl")).open("x", encoding="utf-8")
            )
            for scenario in scenarios
        ]
        for response in flow.run_batch() if batch else flow.run():
            processed += 1
            raw.write(json.dumps(response, ensure_ascii=False) + "\n")
            if response.get("error"):
                errors += 1
                continue
            for index, handle in enumerate(handles):
                formatted = flow.format_response(
                    original, response, scenarios[index], flow.ac_type, index
                )
                if formatted:
                    accepted += 1
                    handle.write(json.dumps(formatted, ensure_ascii=False) + "\n")
    receipt = {
        "processed": processed,
        "accepted_conversations": accepted,
        "errors": errors,
        "seed": seed,
        "batch": batch,
        "historical_reproduction": False,
    }
    with (output / "receipt.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clarity")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "stats"]:
        command = commands.add_parser(name, help="Check and summarize released benchmark records")
        command.add_argument("--data-root", type=Path, default=Path("data"))
        command.add_argument("--dataset", choices=DATASETS)
    command = commands.add_parser("generate", help="Generate new records using your model endpoint")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed", type=int, default=10)
    command.add_argument("--batch", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in {"validate", "stats"}:
            result = [
                validate_dataset(args.data_root, dataset)
                for dataset in ([args.dataset] if args.dataset else DATASETS)
            ]
            print(json.dumps(result, indent=2))
            return 0
        result = generate(args.config, args.output, args.seed, args.batch)
        print(json.dumps(result, indent=2))
        return 1 if result["errors"] else 0
    except (ValueError, OSError, KeyError) as error:
        print(f"clarity: {error}", file=sys.stderr)
        return 1
