import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase

logger = logging.getLogger(LOGGER_NAME)


class RetrieverDecision(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> bool:

        final_decision = False
        if state.retriever_result:
            ac_terms = [
                list(set(state.retriever_result[m].group) - set(state.example.cois[m]))
                for m in state.retriever_result
            ]
            ac_terms = [len(ac_terms[m]) >= 1 for m in state.retriever_result]
            state.example.retriever_decision = ac_terms
            if all(ac_terms):
                final_decision = True
        return final_decision
