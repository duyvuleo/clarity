import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME, SettingKey
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase

logger = logging.getLogger(LOGGER_NAME)


class ParserDecision(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> bool:
        flow_config = self._get_flow_config(config)
        final_decision = True

        if (
            state.parser_result
            and ("selected" in state.parser_result)
            and state.parser_result["selected"] != "null"
        ) and state.retriever_result:
            if (
                SettingKey.COLUMN_AMBIGUITY == flow_config["llm"].ac_type
                or SettingKey.COLUMN_CONFUSION == flow_config["llm"].ac_type
            ):
                query_columns = state.parser_result["columns"]
            else:
                query_columns = state.parser_result["value_pairs"]
            # if there are no available columns left in the coming iteration - valid for cases where only one column name is mentioned
            if not (
                len(query_columns) >= 2
                or (
                    len(query_columns) == 1
                    and isinstance(next(iter(query_columns.values())), list)
                    and len(next(iter(query_columns.values()))) >= 1
                )
            ):
                final_decision = False

        if (not state.parser_result) or (
            ("selected" in state.parser_result) and (state.parser_result["selected"] == "null")
        ):
            final_decision = False
        return final_decision
