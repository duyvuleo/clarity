import logging
from collections import Counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.signatures import UtteranceAggregatorResponse
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class UtteranceResponseAggregator(FlowNodeBase):
    def __init__(self, output_key: str = "utt_agg_result"):
        self.output_key = output_key

    @staticmethod
    def majority_vote(all_outcomes, all_reason):

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

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))

        response = {}
        if state.utt_eval_result:
            outcomes = [state.utt_eval_result[i].consistency_outcome for i in state.utt_eval_result]
            reasons = [state.utt_eval_result[i].consistency_reason for i in state.utt_eval_result]
            consistency_vote, consistency_reason = UtteranceResponseAggregator.majority_vote(
                outcomes, reasons
            )

            outcomes = [state.utt_eval_result[i].ac_outcome for i in state.utt_eval_result]
            reasons = [state.utt_eval_result[i].ac_reason for i in state.utt_eval_result]
            ac_vote, ac_reason = UtteranceResponseAggregator.majority_vote(outcomes, reasons)

            response = UtteranceAggregatorResponse(
                consistency_outcome=consistency_vote,
                consistency_reason=consistency_reason,
                ac_outcome=ac_vote,
                ac_reason=ac_reason,
            )

        return {self.output_key: response}
