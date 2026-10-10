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
from clarity.generation.task_flows.components.signatures import TermEvaluatorResponse


from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ACTermEvaluator(UtteranceNodeBase):
    def __init__(
        self,
        output_pydantic_object: BaseModel,
        input_fields: list[str] = [
            "dataset_columns",
            "term",
            "cols_to_compare",
            "coi",
            "coi_value",
            "vals_to_compare",
            "reason",
        ],
        output_key: str = "term_eval_result",
    ):
        super().__init__(
            output_pydantic_object=output_pydantic_object,
            input_fields=input_fields,
            output_key=output_key,
        )

    def LLM_call(self, state, flow_config, prompt_key, judge):

        response = self._generate(
            example=state.example,
            llm=flow_config.llm_eval_term[judge],
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
        logger.info(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)

        output, response = {}, {}

        for m in range(flow_config.multifacet):
            output[m] = {}

            # This is to check if one of the multi-facets is valid
            iterative_check = True
            if state.term_agg_result:
                if (
                    state.term_agg_result[m].similarity_outcome.lower() == "valid"
                    and state.term_agg_result[m].ac_outcome.lower() == "valid"
                ):
                    iterative_check = False
                    output[m] = state.term_eval_result[m]

            if state.term_gen_result and iterative_check:
                state.example.term = state.term_gen_result[m].term
                # setting under different ac use cases
                if flow_config.ac_type == SettingKey.COLUMN_AMBIGUITY:
                    state.example.ac_group = state.term_gen_result[m].group
                    state.example.cols_to_compare = list(
                        set(state.example.dataset_columns)
                        - set(state.example.ac_group)
                        - {state.example.cois[m]}
                    )  # add[m] when there are multiple columns

                    if flow_config.similarity_criteria == SettingKey.LEXICAL_SIMILARITY:
                        prompt_similarity = UtterancePromptConfigKey.LEXICAL_SIMILARITY
                        prompt_key = UtterancePromptConfigKey.LEXICAL_AMBIGUITY
                    else:
                        prompt_similarity = UtterancePromptConfigKey.SEMANTIC_SIMILARITY
                        prompt_key = UtterancePromptConfigKey.SEMANTIC_AMBIGUITY

                elif flow_config.ac_type == SettingKey.COLUMN_CONFUSION:
                    prompt_key = UtterancePromptConfigKey.CONFUSION_TERM_VALIDITY

                elif flow_config.ac_type == SettingKey.VALUE_CONFUSION:
                    state.example.coi = state.example.cois[m]
                    state.example.coi_value = state.example.column_values[m]
                    prompt_key = UtterancePromptConfigKey.CONFUSION_VALUE_VALIDITY

                elif flow_config.ac_type in (
                    SettingKey.VALUE_AMBIGUITY_SIGL_COL,
                    SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
                ):
                    state.example.ac_group = state.term_gen_result[m].group
                    state.example.reason = state.term_gen_result[m].reason
                    if flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL:
                        state.example.vals_to_compare = state.example.subset_columns
                        prompt_key = (
                            UtterancePromptConfigKey.SINGLE_COL_VALUE_TERM_SIMILARITY_EVAL_PROMPT
                        )
                    elif flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_MULTI_COLS:
                        state.example.cols_to_compare = state.example.ac_group
                        prompt_key = (
                            UtterancePromptConfigKey.MULTI_COLS_VALUE_TERM_SIMILARITY_EVAL_PROMPT
                        )
                    # input_fields = self.input_fields + ['voi', 'subset_columns']
                    # response = self.LLM_call(state, flow_config, prompt_key, output_pydantic_object, input_fields=self.input_fields)

                for judge in range(len(flow_config.llm_eval_term)):
                    # ambiguity or confusion check
                    ac_check = self.LLM_call(state, flow_config, prompt_key, judge)

                    # only for lexical cases
                    if flow_config.ac_type == SettingKey.COLUMN_AMBIGUITY:
                        # similarity check
                        state.example.cols_to_compare = state.example.ac_group + [
                            state.example.cois[m]
                        ]
                        similarity = self.LLM_call(state, flow_config, prompt_similarity, judge)

                        ##dissimilarity check
                        if flow_config.similarity_criteria == SettingKey.LEXICAL_SIMILARITY:
                            dissimilarity = self.LLM_call(
                                state,
                                flow_config,
                                UtterancePromptConfigKey.LEXICAL_DISSIMILARITY,
                                judge,
                            )
                        else:
                            dissimilarity = self.output_pydantic_object()
                    elif flow_config.similarity_criteria == SettingKey.COLUMN_CONFUSION:
                        dissimilarity = self.LLM_call(
                            state,
                            flow_config,
                            UtterancePromptConfigKey.LEXICAL_DISSIMILARITY,
                            judge,
                        )
                        similarity = self.output_pydantic_object()
                    else:
                        similarity = self.output_pydantic_object()
                        dissimilarity = self.output_pydantic_object()

                    if (
                        ac_check.outcome.lower() != "null"
                        and similarity.outcome.lower() != "null"
                        and dissimilarity.outcome.lower() != "null"
                        and flow_config.ac_type == SettingKey.COLUMN_AMBIGUITY
                        and SettingKey.LEXICAL_SIMILARITY
                    ):
                        response = TermEvaluatorResponse(
                            similarity_outcome=similarity.outcome,
                            similarity_reason=similarity.reason,
                            dissimilarity_outcome=dissimilarity.outcome,
                            dissimilarity_reason=dissimilarity.reason,
                            ac_outcome=ac_check.outcome,
                            ac_reason=ac_check.reason,
                        )

                    elif (
                        ac_check.outcome.lower() != "null"
                    ):  # and dissimilarity.outcome.lower()!='null':
                        response = TermEvaluatorResponse(
                            similarity_outcome=similarity.outcome,
                            similarity_reason=similarity.reason,
                            dissimilarity_outcome=dissimilarity.outcome,
                            dissimilarity_reason=dissimilarity.reason,
                            ac_outcome=ac_check.outcome,
                            ac_reason=ac_check.reason,
                        )

                    output[m][judge] = response
        return {self.output_key: output}
