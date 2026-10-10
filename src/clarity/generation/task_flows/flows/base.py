"""Load portable generation tasks without private retrieval services."""

from importlib.resources import files
from pathlib import Path

from clarity.core.flows.base import BaseFlow
from clarity.datasets import dataset_directory
from clarity.generation.data_loader import UtteranceDataLoader
from clarity.generation.task_flows.components.data_types import ConversationGeneratorTask
from clarity.models import Llm
from clarity.utils.file_io import load_json


class UtteranceFlowBase(BaseFlow):
    def load_task(self, scenario):
        path = Path(self._task).resolve()
        config = load_json(path)
        task = ConversationGeneratorTask.model_validate(config)
        dataset = config.get("dataset")
        root = (path.parent / config.get("data_root", "../data")).resolve()
        directory = dataset_directory(root, dataset)
        task.train_data_path = str(directory / "seeds.jsonl")
        task.schema_file = str(directory / "schemas.json")
        task.column_file = str(directory / "columns.json")
        task.value_file = str(directory / "values.json.gz")
        task.prompt_config_path = str(files("clarity").joinpath("prompts/generation.yaml"))
        schemas = load_json(task.schema_file)
        columns = load_json(task.column_file)
        values = load_json(task.value_file) if "value" in task.ac_type else {}
        examples = UtteranceDataLoader()(task.train_data_path, scenario, schemas, columns, values)
        if config.get("max_examples") is not None:
            examples = examples[: config["max_examples"]]
        task.train_data = examples
        if task.ac_type == "column_ambiguity" and task.retrieval_type != "llm":
            from clarity.retrieval.factory import build_schema_index

            task.schema_index = build_schema_index(
                task.schema_retrieval, task.retrieval_type, schemas, columns, path.parent
            )
        for field in ["llm_retrieval", "llm_eval_utt", "llm_eval_term", "llm_sample_selectors"]:
            if (
                field == "llm_retrieval"
                and task.retrieval_type != "llm"
                and not task.schema_retrieval.llm_filter
            ):
                task.llm_retrieval = []
                continue
            setattr(task, field, [Llm.from_params(options) for options in getattr(task, field)])
        task.llm_gen_utt = Llm.from_params(task.llm_gen_utt)
        task.llm_gen_term = Llm.from_params(task.llm_gen_term)
        return task

    def print_flow(self):
        return self.flow.get_graph().draw_mermaid()
