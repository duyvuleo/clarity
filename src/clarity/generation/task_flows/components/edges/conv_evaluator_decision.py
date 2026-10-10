import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase

logger = logging.getLogger(LOGGER_NAME)


class ConvEvaluatorDecision(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> bool:
        # sample implementation commnted and true bein returned for now
        """
        final_decision = False
        if state.conv_agg_result is not None:
            temp = state.conv_agg_result.outcome
            final_decision = all(
                [temp[key]["majority_vote"] == "valid" for key in temp.keys()]
            )
            if final_decision is False:
                temp_feedback = [
                    temp[key]["feedback"]
                    for key in temp.keys()
                    if temp[key]["majority_vote"] == "invalid"
                ]
                feedback = " ".join(temp_feedback)
                state.example.history_conv = feedback
        """
        final_decision = True
        return final_decision
