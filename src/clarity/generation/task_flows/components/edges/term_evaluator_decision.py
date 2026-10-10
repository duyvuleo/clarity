import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME, SettingKey
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase

logger = logging.getLogger(LOGGER_NAME)


class TermEvaluatorDecision(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> bool:
        flow_config = self._get_flow_config(config)
        final_decision = True
        #'m' is for multi-facet use case
        if state.term_agg_result:
            state.example.iteration += 1

            if (
                flow_config["llm"].ac_type == SettingKey.COLUMN_AMBIGUITY
            ):  # flow_config['llm'].similarity_criteria == SettingKey.LEXICAL_SIMILARITY and
                all_results = [
                    [
                        state.term_agg_result[m].similarity_outcome.lower(),
                        state.term_agg_result[m].ac_outcome.lower(),
                    ]
                    for m in state.term_agg_result
                ]
            else:
                all_results = [
                    [state.term_agg_result[m].ac_outcome.lower()] for m in state.term_agg_result
                ]

            if any("invalid" in sublist for sublist in all_results):
                final_decision = False
        if state.example.iteration > 2 and final_decision == False:
            final_decision = True
            state.example.iteration = -1

        # this is specifically linked with quality control, where the ambiguous term is linked with many columns beyond the selected group
        # in such cases don't generate the ac utterance
        if final_decision == True:
            state.example.iteration = (
                state.example.iteration if state.example.iteration == -1 else 0
            )
            quality_check = [
                state.term_agg_result[m].dissimilarity_outcome.lower()
                for m in state.term_agg_result
            ]
            if any(elem == "invalid" for elem in quality_check):
                state.example.iteration = -1

        return final_decision
