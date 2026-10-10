import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import (
    LOGGER_NAME,
    UtterancePromptConfigKey,
    SettingKey,
)
from clarity.generation.task_flows.components.nodes.base import UtteranceNodeBase
from clarity.generation.task_flows.components.states import ConversationState
from clarity.generation.task_flows.components.signatures import (
    TCClassificationResponse,
    TCCheckACResponse,
    TCVerifyACResponse,
)
from collections import Counter

from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class SampleSelector(UtteranceNodeBase):
    def __init__(
        self,
        input_fields: list[str] = ["db_schema", "cols_to_compare", "term", "ac_query"],
        output_key: str = "sample_selector",
    ):
        super().__init__(
            input_fields=input_fields,
            output_key=output_key,
        )

    def LLM_call(self, state, flow_config, prompt_key, response_template, judge):

        response = self._generate(
            example=state.example,
            llm=flow_config.llm_sample_selectors[judge],
            meta_prompts=flow_config.meta_prompts,
            output_pydantic_object=response_template,
            prompt_key=prompt_key,
            input_fields=self.input_fields,
        )

        if isinstance(response, str) and ERROR_MESSAGE_TAG in response:
            eval_res = response_template()
            eval_res.reason = response
            response = eval_res

        return response

    @staticmethod
    def test_validity(classify, detection):
        response = False
        if classify.lower() == "untranslatable" and detection.lower() == "yes":
            response = True
        return response

    @staticmethod
    def majority_vote(all_outcomes):
        outcomes = list(all_outcomes)
        # Empty votes, ties and non-boolean outcomes reject the sample.
        return (
            bool(outcomes)
            and all(type(value) is bool for value in outcomes)
            and sum(outcomes) > len(outcomes) / 2
        )

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)
        final_outcome = False
        classify_response, detect_ac_response, verify_ac_response, outcome = {}, {}, {}, {}

        if state.conv_gen_result and state.conv_gen_result.conversation != "null":
            if flow_config.ac_type != SettingKey.VALUE_CONFUSION:
                for judge in range(len(flow_config.llm_sample_selectors)):
                    classify_response[judge] = self.LLM_call(
                        state,
                        flow_config,
                        UtterancePromptConfigKey.TC1,
                        TCClassificationResponse,
                        judge,
                    )

                    detect_ac_response[judge] = self.LLM_call(
                        state, flow_config, UtterancePromptConfigKey.TC2_1, TCCheckACResponse, judge
                    )

                    state.example.cols_to_compare = (
                        detect_ac_response[judge].terms
                        if detect_ac_response[judge].terms
                        else ["None found"]
                    )
                    verify_ac_response[judge] = self.LLM_call(
                        state,
                        flow_config,
                        UtterancePromptConfigKey.TC2_2,
                        TCVerifyACResponse,
                        judge,
                    )

                    outcome[judge] = SampleSelector.test_validity(
                        classify_response[judge].category, verify_ac_response[judge].outcome
                    )

                final_outcome = SampleSelector.majority_vote(outcome.values())
            else:
                final_outcome = True

        return {
            self.output_key: {
                "outcome": final_outcome,
                "detect_ac": detect_ac_response,
                "verify_ac": verify_ac_response,
            }
        }
