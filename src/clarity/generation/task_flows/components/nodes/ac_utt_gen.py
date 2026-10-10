import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import (
    UtterancePromptConfigKey,
    LOGGER_NAME,
    SettingKey,
)
from clarity.generation.task_flows.components.nodes.base import UtteranceNodeBase
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ACUtteranceGenerator(UtteranceNodeBase):
    def __init__(
        self,
        prompt_key: str,
        output_pydantic_object: BaseModel,
        input_fields: list[str] = [
            "expected_query",
            "cois",
            "term",
            "history_conv",
            "question",
            "coi",
            "cols_to_compare",
            "voi",
        ],
        output_key: str = "utt_gen_result",
    ):
        super().__init__(
            prompt_key=prompt_key,
            output_pydantic_object=output_pydantic_object,
            input_fields=input_fields,
            output_key=output_key,
        )

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)

        response = {}

        if state.utt_agg_result:
            state.example.history_conv = " ".join(
                [(state.example.history_conv or ""), state.utt_agg_result["consistency_reason"]]
            )

        if state.term_gen_result and state.example.iteration != -1:
            state.example.iteration = (
                0 if state.example.iteration == -1 else state.example.iteration
            )
            prompt_key = self.prompt_key

            state.example.term = [
                state.term_gen_result[m].term for m in range(flow_config.multifacet)
            ]
            # prompts value cases are different
            if flow_config.ac_type == SettingKey.VALUE_CONFUSION:
                prompt_key = UtterancePromptConfigKey.UTT_GENERATOR_VALUE
                state.example.coi = [item[0] for item in state.example.cois]

            if flow_config.ac_type in (
                SettingKey.VALUE_AMBIGUITY_SIGL_COL,
                SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
            ):
                state.example.coi = [item[0] for item in state.example.cois]
                if flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL:
                    prompt_key = UtterancePromptConfigKey.UTT_SINGLE_COL_VALUE_AMBIGUITY_GENERATOR
                else:
                    prompt_key = UtterancePromptConfigKey.UTT_MULTI_COLS_VALUE_AMBIGUITY_GENERATOR

            response = self._generate(
                example=state.example,
                llm=flow_config.llm_gen_utt,
                meta_prompts=flow_config.meta_prompts,
                output_pydantic_object=self.output_pydantic_object,
                prompt_key=prompt_key,
                input_fields=self.input_fields,
            )

            if isinstance(response, str) and ERROR_MESSAGE_TAG in response:
                gen_res = self.output_pydantic_object()
                gen_res.nl_query = response
                response = gen_res
            else:
                state.example.new_query = response.nl_query
                state.example.ac_query = response.ac_query
            logger.info(f"Original Utterance: {response.nl_query}")
            logger.info(f"Generated Utterance: {response.ac_query}")
        return {self.output_key: response}
