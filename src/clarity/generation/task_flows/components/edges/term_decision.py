import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase

logger = logging.getLogger(LOGGER_NAME)


class ACTermDecision(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> bool:

        final_decision = False

        if state.term_gen_result:
            all_results = [state.term_gen_result[m].term for m in state.term_gen_result]
            decisions = [
                (result.lower() != "null" and result.lower() != "none") for result in all_results
            ]

            state.example.term_decision = decisions
            if all(decisions):
                final_decision = True

        return final_decision
