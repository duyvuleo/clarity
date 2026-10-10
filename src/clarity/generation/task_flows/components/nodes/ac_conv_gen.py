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

from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ACConversationGenerator(UtteranceNodeBase):
    def __init__(
        self,
        output_pydantic_object: BaseModel,
        input_fields: list[str] = [
            "type",
            "conversation",
            "cois",
            "expected_query",
            "term",
            "column_values",
        ],
        output_key: str = "conv_gen_result",
    ):
        super().__init__(
            output_pydantic_object=output_pydantic_object,
            input_fields=input_fields,
            output_key=output_key,
        )

    @staticmethod
    def remove_duplicates(strings: str):
        seen = set()
        result = []
        for s in strings:
            lower_s = s.lower()
            if lower_s not in seen:
                seen.add(lower_s)
                result.append(s)
        return result

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)
        template_ambiguous = "We noticed that the term '{}' might also refer to the {} in the data. We can use one of these in the results instead if you'd like"
        template_confusion_column = "I can't help you with your request because I can't find {} in this database, and there are no close matches."
        template_confusion_value = "I can't help you with your request because I can't find {} against {} column in the database."

        response = {}

        if state.utt_gen_result and state.example.iteration != -1:
            if flow_config.ac_type in [
                SettingKey.COLUMN_AMBIGUITY,
                SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
                SettingKey.VALUE_AMBIGUITY_SIGL_COL,
            ]:
                if flow_config.multifacet == 1:
                    if flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL:
                        state.example.ac_group = ACConversationGenerator.remove_duplicates(
                            [state.example.cois[0][1]] + state.term_gen_result[0].group
                        )
                        columns = list(set(state.example.cois + state.example.ac_group))
                    elif flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_MULTI_COLS:
                        state.example.ac_group = ACConversationGenerator.remove_duplicates(
                            state.term_gen_result[0].group
                        )
                        columns = state.example.cols_to_compare
                    else:
                        state.example.ac_group = ACConversationGenerator.remove_duplicates(
                            state.example.cois + state.term_gen_result[0].group
                        )
                        columns = list(set(state.example.cois + state.example.ac_group))
                    columns = [f"'{v}'" for v in columns]
                    template_response = ", ".join(columns[:-1]) + f", or {columns[-1]}"

                    template_response = template_ambiguous.format(
                        state.example.term[0], template_response
                    )
                    state.example.conversation = [
                        {"role": "USER", "content": state.example.ac_query},
                        {"role": "AGENT", "content": template_response},
                    ]
                    prompt_key = UtterancePromptConfigKey.CONV_GENERATOR_SINGLE

                else:
                    if flow_config.ac_type in [
                        SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
                        SettingKey.VALUE_AMBIGUITY_SIGL_COL,
                    ]:
                        state.example.ac_group = [
                            ACConversationGenerator.remove_duplicates(
                                [state.example.cois[m][1]] + state.term_gen_result[m].group
                            )
                            for m in range(flow_config.multifacet)
                        ]
                    else:
                        state.example.ac_group = [
                            ACConversationGenerator.remove_duplicates(
                                [state.example.cois[m]] + state.term_gen_result[m].group
                            )
                            for m in range(flow_config.multifacet)
                        ]
                    template_response = [{"role": "USER", "content": state.example.ac_query}]
                    user_response = [f"'{v}'" for v in state.example.cois]
                    user_response = ", ".join(user_response)
                    agent_response = []
                    for m in range(flow_config.multifacet):
                        columns = list(set([state.example.cois[m]] + state.example.ac_group[m]))
                        columns = [f"'{v}'" for v in columns]
                        temp_response = ", ".join(columns[:-1]) + f", or {columns[-1]}"
                        agent_response.append(
                            template_ambiguous.format(state.example.term[m], temp_response)
                        )

                    agent_response = ". ".join(agent_response)
                    template_response.append({"role": "AGENT", "content": agent_response})
                    template_response.append({"role": "USER", "content": user_response})

                    state.example.conversation = template_response
                    prompt_key = UtterancePromptConfigKey.CONV_GENERATOR_MULTI

            elif (
                flow_config.ac_type == SettingKey.COLUMN_CONFUSION
                or flow_config.ac_type == SettingKey.VALUE_CONFUSION
            ):
                terms = (
                    state.example.term
                    if len(state.example.term) == 0
                    else ", ".join(state.example.term[:-1]) + " and " + state.example.term[-1]
                )

                if flow_config.ac_type == SettingKey.COLUMN_CONFUSION:
                    prompt_key = UtterancePromptConfigKey.CONV_GENERATOR_CONFUSION_COLUMN
                    response = template_confusion_column.format(terms)
                else:
                    prompt_key = UtterancePromptConfigKey.CONV_GENERATOR_CONFUSION_VALUE
                    state.example.coi = [k[0] for k in state.example.cois]
                    response = template_confusion_value.format(terms, state.example.coi)

                state.example.conversation = [
                    {"role": "USER", "content": state.example.ac_query},
                    {"role": "AGENT", "content": response},
                ]
            final_conv = []
            for conv_type in [
                "concise helpful",
                "verbose helpful",
                "partially helpful",
                "not helpful - no column",
                "not helpful - random column",
            ]:
                state.example.type = conv_type

                response = self._generate(
                    example=state.example,
                    llm=flow_config.llm_gen_term,
                    meta_prompts=flow_config.meta_prompts,
                    output_pydantic_object=self.output_pydantic_object,
                    prompt_key=prompt_key,
                    input_fields=self.input_fields,
                )

                if isinstance(response, str) and ERROR_MESSAGE_TAG in response:
                    gen_res = self.output_pydantic_object()
                    gen_res.conversation = response
                    response = gen_res
                else:
                    final_conv.append(response.conversation)
                    if "confusion" in flow_config.ac_type:
                        break
            state.example.conversation = final_conv
            logger.info(Color.RED(f"Generated conversation: {response.conversation}"))

        return {self.output_key: response}
