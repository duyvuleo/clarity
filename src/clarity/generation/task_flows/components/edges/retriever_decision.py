import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase

logger = logging.getLogger(LOGGER_NAME)


class RetrieverDecision(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> bool:

        from clarity.generation.task_flows.components.data_types import SettingKey

        task = self._get_flow_config(config)["llm"]
        decisions = []
        for index, result in (state.retriever_result or {}).items():
            group = result.group
            if not isinstance(group, list):
                decisions.append(False)
                continue
            target = state.example.cois[index]
            if isinstance(target, (tuple, list)):
                target = (
                    target[1] if task.ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL else target[0]
                )
            candidates = {str(value).strip().casefold() for value in group}
            decisions.append(bool(candidates - {str(target).strip().casefold()}))
        state.example.retriever_decision = decisions
        return bool(decisions) and all(decisions)
