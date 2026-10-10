import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import (
    LOGGER_NAME,
    SettingKey,
    UtterancePromptConfigKey,
)
from clarity.generation.task_flows.components.nodes.base import UtteranceNodeBase
from clarity.generation.task_flows.components.states import ConversationState
from clarity.generation.task_flows.components.signatures import UtteranceEvaluatorResponse

from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ACUtteranceEvaluator(UtteranceNodeBase):
    def __init__(
        self,
        output_pydantic_object: BaseModel,
        input_fields: list[str] = [
            "ac_query",
            "cois",
            "coi",
            "expected_query",
            "term",
            "cols_to_compare",
            "vals_to_compare",
            "dataset_columns",
            "question",
            "column_values",
        ],
        output_key: str = "utt_eval_result",
    ):
        super().__init__(
            output_pydantic_object=output_pydantic_object,
            input_fields=input_fields,
            output_key=output_key,
        )

    def LLM_call(self, state, flow_config, prompt_key, judge):

        response = self._generate(
            example=state.example,
            llm=flow_config.llm_eval_utt[judge],
            meta_prompts=flow_config.meta_prompts,
            output_pydantic_object=self.output_pydantic_object,
            prompt_key=prompt_key,
            input_fields=self.input_fields,
        )

        if isinstance(response, str) and ERROR_MESSAGE_TAG in response:
            eval_res = self.output_pydantic_object()
            eval_res.reason = response
            response = eval_res

        return response

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)
        output = {}

        if state.utt_gen_result:
            for judge in range(len(flow_config.llm_eval_utt)):
                if flow_config.ac_type == SettingKey.COLUMN_AMBIGUITY:
                    # coi might will need to be updated for the multi-facted case
                    state.example.cols_to_compare = [
                        list(
                            dict.fromkeys(state.term_gen_result[m].group + [state.example.cois[m]])
                        )
                        for m in range(flow_config.multifacet)
                    ]

                    # ambiguity check
                    amb_prompt = (
                        UtterancePromptConfigKey.UTT_SEMANTIC_AMBIGUITY
                        if flow_config.similarity_criteria == SettingKey.SEMANTIC_SIMILARITY
                        else UtterancePromptConfigKey.UTT_LEXICAL_AMBIGUITY
                    )
                    ac = self.LLM_call(state, flow_config, amb_prompt, judge)

                    # consistency check
                    consistency = self.LLM_call(
                        state,
                        flow_config,
                        UtterancePromptConfigKey.UTT_CONSISTENCY_AMBIGUITY,
                        judge,
                    )

                elif (
                    flow_config.ac_type == SettingKey.COLUMN_CONFUSION
                    or flow_config.ac_type == SettingKey.VALUE_CONFUSION
                ):
                    if flow_config.ac_type == SettingKey.COLUMN_CONFUSION:
                        ac_prompt = UtterancePromptConfigKey.UTT_CONFUSION_COLUMN
                        state.example.cols_to_compare = state.example.cois
                    else:
                        ac_prompt = UtterancePromptConfigKey.UTT_CONFUSION_VALUE
                        state.example.cols_to_compare = state.example.cois_pair
                    # confusion check
                    ac = self.LLM_call(state, flow_config, ac_prompt, judge)

                    # consistency check
                    consistency = self.LLM_call(
                        state,
                        flow_config,
                        UtterancePromptConfigKey.UTT_CONSISTENCY_CONFUSION,
                        judge,
                    )
                elif flow_config.ac_type in (
                    SettingKey.VALUE_AMBIGUITY_SIGL_COL,
                    SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
                ):
                    # ambiguity check
                    if flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL:
                        amb_prompt = UtterancePromptConfigKey.UTT_SINGLE_COL_VALUE_AMBIGUITY
                    else:
                        amb_prompt = UtterancePromptConfigKey.UTT_MULTI_COLS_VALUE_AMBIGUITY

                    ac = self.LLM_call(state, flow_config, amb_prompt, judge)

                    # consistency check
                    consistency = self.LLM_call(
                        state, flow_config, UtterancePromptConfigKey.UTT_CONSISTENCY_AMB, judge
                    )

                else:
                    raise ValueError(
                        f"Please implement utterance evaluator for ac_type: {flow_config.ac_type}!"
                    )

                response = UtteranceEvaluatorResponse(
                    consistency_outcome=consistency.outcome,
                    consistency_reason=consistency.reason,
                    ac_outcome=ac.outcome,
                    ac_reason=ac.reason,
                )
                output[judge] = response
        logger.info(Color.RED(f"[Node]: ## {self.__class__.__name__} completed ##"))
        return {self.output_key: output}
