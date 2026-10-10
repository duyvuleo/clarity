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
    ConfusionTermResponse,
    ConfusionValueResponse,
    AmbiguityValueTermResponse,
    AmbiguityTermResponse,
)
from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ACTermGenerator(UtteranceNodeBase):
    def __init__(
        self,
        output_pydantic_object: BaseModel,
        input_fields: list[str] = [
            "coi",
            "subset_columns",
            "history_term",
            "question",
            "dataset_columns",
            "coi_value",
        ],
        output_key: str = "term_gen_result",
    ):
        super().__init__(
            output_pydantic_object=output_pydantic_object,
            input_fields=input_fields,
            output_key=output_key,
        )

    def LLM_call(self, state, flow_config, prompt_key, output_pydantic_object, input_fields=None):
        response = self._generate(
            example=state.example,
            llm=flow_config.llm_gen_term,
            meta_prompts=flow_config.meta_prompts,
            output_pydantic_object=output_pydantic_object,
            prompt_key=prompt_key,
            input_fields=self.input_fields if not input_fields else input_fields,
        )

        if isinstance(response, str) and ERROR_MESSAGE_TAG in response:
            gen_res = self.output_pydantic_object()
            gen_res.reason = response
            response = gen_res

        return response

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)
        output, response = {}, {}

        iterative_check = True  # if ac term needs to be fixed in the next round or iteration

        for m in range(flow_config.multifacet):
            eval_check = True
            iteration_count = 0

            if state.example.db_id is not None:
                key = next(iter(state.parser_result["selected"][m]))
                state.example.dataset_columns = [
                    col.lower() for col in state.example.spider_columns[key]
                ]
                state.example.db_schema = state.example.spider_schema[key]

            ## iterating to fix ac_term that exactly matches with the db schema columns
            while eval_check and iteration_count < 3:
                if state.term_agg_result:
                    ##check for history for correcting any errors
                    if (
                        flow_config.ac_type == SettingKey.COLUMN_AMBIGUITY
                    ):  # and flow_config.similarity_criteria == SettingKey.LEXICAL_SIMILARITY:
                        # to check for each 'm' of multifacet
                        if (
                            state.term_agg_result[m].similarity_outcome.lower() == "valid"
                            and state.term_agg_result[m].ac_outcome.lower() == "valid"
                        ):
                            iterative_check, eval_check = False, False
                            output[m] = state.term_gen_result[m]
                        else:
                            state.example.history_term = " ".join(
                                [
                                    (state.example.history_term or ""),
                                    state.term_agg_result[m].similarity_reason,
                                    state.term_agg_result[m].ac_reason,
                                ]
                            )
                    # this is for column confusion cases
                    else:
                        if state.term_agg_result[m].ac_outcome.lower() == "valid":
                            iterative_check, eval_check = False, False
                            output[m] = state.term_gen_result[m]
                        else:
                            state.example.history_term = " ".join(
                                [
                                    (state.example.history_term or ""),
                                    state.term_agg_result[m].ac_reason,
                                ]
                            )

                # if retriever was successful and setting is column ambiguity
                if (
                    state.retriever_result
                    and iterative_check
                    and flow_config.ac_type == SettingKey.COLUMN_AMBIGUITY
                ):
                    state.example.coi = state.retriever_result[m].coi
                    state.example.subset_columns = state.retriever_result[m].group
                    prompt_key = (
                        UtterancePromptConfigKey.SEMANTIC_TERM
                        if flow_config.similarity_criteria == SettingKey.SEMANTIC_SIMILARITY
                        else UtterancePromptConfigKey.LEXICAL_TERM
                    )
                    output_pydantic_object = self.output_pydantic_object
                    response = self.LLM_call(state, flow_config, prompt_key, output_pydantic_object)

                # column confusion setting
                elif iterative_check and flow_config.ac_type == SettingKey.COLUMN_CONFUSION:
                    state.example.coi = state.example.cois[m]
                    output_pydantic_object = ConfusionTermResponse
                    response = self.LLM_call(
                        state,
                        flow_config,
                        UtterancePromptConfigKey.CONFUSION_TERM,
                        output_pydantic_object,
                    )

                # value ambiguity settings
                elif flow_config.ac_type in (
                    SettingKey.VALUE_AMBIGUITY_SIGL_COL,
                    SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
                ):
                    if flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL:
                        # For a value in a single column, we generate an ambiguity term that could refer to more than one value in that column
                        prompt_key = UtterancePromptConfigKey.SINGLE_COL_VALUE_AMBIGUITY
                        output_pydantic_object = AmbiguityValueTermResponse
                    elif flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_MULTI_COLS:
                        # For a value in multiple columns, we generate an ambiguity term that could refer to more than two of those columns
                        prompt_key = UtterancePromptConfigKey.MULTI_COLS_VALUE_AMBIGUITY
                        output_pydantic_object = AmbiguityTermResponse
                    input_fields = self.input_fields + ["voi", "subset_columns"]
                    response = self.LLM_call(
                        state,
                        flow_config,
                        prompt_key,
                        output_pydantic_object,
                        input_fields=input_fields,
                    )

                # value confusion setting
                elif iterative_check and flow_config.ac_type == SettingKey.VALUE_CONFUSION:
                    state.example.coi = state.example.cois[m][0]
                    state.example.coi_value = state.example.column_values[m]

                    prompt_key = UtterancePromptConfigKey.CONFUSION_VALUE
                    output_pydantic_object = ConfusionValueResponse

                    # this is specifically for large coi values as it's not feasible to inject 1000s of values to the prompt
                    response = (
                        {}
                        if len(state.example.coi_value) > 25
                        else self.LLM_call(state, flow_config, prompt_key, output_pydantic_object)
                    )

                if flow_config.ac_type != SettingKey.VALUE_CONFUSION and hasattr(response, "term"):
                    # to check if the ac term is an exact match with columns in the schema
                    if response.term.lower() != "null":
                        eval_check = any(
                            [
                                response.term.lower() == item.lower()
                                for item in state.example.dataset_columns
                            ]
                        )

                    if response.term.lower() != "null" and eval_check:
                        state.example.history_term = (
                            (state.example.history_term or "")
                            + response.term
                            + " is not a valid term for the given list of columns\n"
                        )
                else:
                    eval_check = False

                iteration_count += 1

                output[m] = response
        # this is linked with column value use case. If there are many values that can't be injected to the prompt then response is not generated
        output = {} if any(not v for v in output.values()) else output

        return {self.output_key: output}
