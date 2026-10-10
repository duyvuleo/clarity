import logging
from typing import Any, Iterable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from clarity.generation.task_flows.components.data_types import (
    LOGGER_NAME,
    UtterancePromptConfigKey,
    SettingKey,
)
from clarity.generation.task_flows.flows.base import UtteranceFlowBase
from tqdm import tqdm

from clarity.core.flows.components.data_types import FlowConfigKey
from clarity.utils.file_io import load_yaml

logger = logging.getLogger(LOGGER_NAME)


class StaticFlow(UtteranceFlowBase):
    def __init__(self, scenario: str, **kwargs) -> None:

        super().__init__(**kwargs)

        task = self.load_task(scenario)
        train_data = task.train_data
        self.data_size = len(train_data)

        self.data_path = task.train_data_path
        self.ac_type = task.ac_type
        task.multifacet = min(task.multifacet, 3)  # to ensure max limit for ac

        prompt_config = load_yaml(task.prompt_config_path)
        task.meta_prompts = prompt_config[UtterancePromptConfigKey.PROMPTS]
        task.dataset_info = {}

        ac_type = SettingKey(task.ac_type)

        self.flow = self.create(ac_type=ac_type)

        self.flow_config = {
            "max_concurrency": getattr(task, "batch_size", 1),
            FlowConfigKey.RECURSION_LIMIT: task.recursion_limit,
            FlowConfigKey.CONFIGURABLE: {
                FlowConfigKey.LLM: task,
            },
        }

        self.flow_inputs = [
            {
                "example": example,
                "parser_result": {},
                "retriever_result": {},
                "retriever_class_result": {},
                "term_gen_result": {},
                "term_eval_result": {},
                "term_agg_result": {},
                "utt_gen_result": {},
                "utt_eval_result": {},
                "utt_agg_result": {},
                "conv_gen_result": {},
                "conv_eval_result": {},
                "conv_agg_result": {},
                "sample_selector": {},
            }
            for example in train_data
        ]

        logger.info("[Train Data]: %d examples", self.data_size)

        # if self.flow_png_path:
        #    self.viz_flow()

        # logger.info('[Flow] Mermaid Draw:')
        # self.print_flow()

    def run_batch(self) -> Iterable[dict[str, Any]]:

        for response in tqdm(
            self.flow.batch_as_completed(
                inputs=self.flow_inputs,
                config=self.flow_config,
                return_exceptions=True,
                output_keys="response",
            ),
            total=self.data_size,
            desc="Processing inputs",
        ):
            response = response[-1]
            if isinstance(response, Exception):
                response = {"error": str(response)}
            else:
                response = response.model_dump(mode="json")
            yield response

    def run(self) -> Iterable[dict[str, Any]]:

        for flow_input in tqdm(self.flow_inputs, desc="Processing inputs"):
            try:
                response = self.flow.invoke(
                    input=flow_input, config=self.flow_config, output_keys="response"
                ).model_dump(mode="json")

            except Exception as e:
                response = {"error": str(e)}
            yield response

    def format_response(
        self, original_data: list, response: dict, scenario: str, ac_type: str, index: int
    ) -> Iterable[dict[str, Any]]:

        if (
            not response.get("conv_gen_result")
            or response.get("sample_selector", {}).get("outcome") is not True
        ):
            return {}
        example = response["example"]
        position = example["example_id"] - 1
        if not 0 <= position < len(original_data):
            raise ValueError("Generated example ID does not match the input seed file")
        record = original_data[position].copy()
        if (record["db_id"], record["question"], record["expectedQuery"]) != (
            example["db_id"],
            example["question"],
            example["expected_query"],
        ):
            raise ValueError("Generated example no longer matches its source seed")
        conversations = example.get("conversation") or []
        if index >= len(conversations) or not conversations[index]:
            return {}
        record.update(
            original_question=example["question"],
            usecase=f"{ac_type}-{scenario}",
            ac_term=example["term"],
            question=example["ac_query"],
            selected_column=example["cois"],
            similar_columns=example["ac_group"],
            conversation=conversations[index],
        )
        return record
