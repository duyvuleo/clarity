import logging
from collections import Counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.signatures import TermAggregatorResponse
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class TermResponseAggregator(FlowNodeBase):
    def __init__(self, output_key: str = "term_agg_result"):
        self.output_key = output_key

    @staticmethod
    def majority_vote(all_outcomes, all_reason):

        # all_outcomes = [evals[i].outcome for i in range(len(evals))]
        # all_reason = [evals[i].reason for i in range(len(evals))]

        vote_count = Counter(all_outcomes)
        most_common = vote_count.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            majority = "invalid"  # tie-breaker defaults to 'invalid'
        else:
            majority = most_common[0][0]

        if majority == "invalid":
            temp_feeback = [
                all_reason[i] for i in range(len(all_outcomes)) if all_outcomes[i] == "invalid"
            ]
        else:
            temp_feeback = [""]

        feedback = " ".join(temp_feeback)

        return majority, feedback

    @staticmethod
    def all_valid(all_outcomes, all_reason):
        temp_feedback = []
        majority = "valid"
        for idx, value in enumerate(all_outcomes):
            if value.lower() == "invalid":
                majority = "invalid"
                temp_feedback.append(all_reason[idx])
        feedback = " ".join(temp_feedback)

        return majority, feedback

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.info(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))

        # sample implementation commnted and valid response being returned
        response = {}
        if state.term_eval_result:
            for m in range(len(state.term_eval_result)):
                outcomes = [
                    state.term_eval_result[m][i].similarity_outcome
                    for i in range(len(state.term_eval_result[m]))
                ]
                reasons = [
                    state.term_eval_result[m][i].similarity_reason
                    for i in range(len(state.term_eval_result[m]))
                ]
                similarity_vote, similarity_reason = TermResponseAggregator.majority_vote(
                    outcomes, reasons
                )

                outcomes = [
                    state.term_eval_result[m][i].dissimilarity_outcome
                    for i in range(len(state.term_eval_result[m]))
                ]
                reasons = [
                    state.term_eval_result[m][i].dissimilarity_reason
                    for i in range(len(state.term_eval_result[m]))
                ]

                dissimilarity_vote, dissimilarity_reason = TermResponseAggregator.majority_vote(
                    outcomes, reasons
                )

                outcomes = [
                    state.term_eval_result[m][i].ac_outcome
                    for i in range(len(state.term_eval_result[m]))
                ]
                reasons = [
                    state.term_eval_result[m][i].ac_reason
                    for i in range(len(state.term_eval_result[m]))
                ]
                ac_vote, ac_reason = TermResponseAggregator.majority_vote(outcomes, reasons)

                response[m] = TermAggregatorResponse(
                    similarity_outcome=similarity_vote,
                    similarity_reason=similarity_reason,
                    dissimilarity_outcome=dissimilarity_vote,
                    dissimilarity_reason=dissimilarity_reason,
                    ac_outcome=ac_vote,
                    ac_reason=ac_reason,
                )

        return {self.output_key: response}
