import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase

logger = logging.getLogger(LOGGER_NAME)


class UtteranceEvaluatorDecision(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> bool:

        final_decision = True

        if state.utt_agg_result:
            state.example.iteration += 1
            # all_results = [state.utt_agg_result['ambiguity_outcome'], state.utt_agg_result['consistency_outcome']]
            if (
                state.utt_agg_result["consistency_outcome"].lower() == "invalid"
            ):  # any(result.lower() == 'invalid' for result in all_results):
                final_decision = False
            if (
                state.utt_agg_result["consistency_outcome"].lower() == "valid"
                and state.utt_agg_result["ac_outcome"].lower() == "invalid"
            ) or (state.example.iteration > 2 and final_decision == False):
                # if state.example.iteration>2 and final_decision == False:
                final_decision = True
                state.example.iteration = -1

        return final_decision
