import logging
from collections import Counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.signatures import ConversationAggregatorResponse
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ConvResponseAggregator(FlowNodeBase):
    def __init__(self, output_key: str = "conv_agg_result"):

        self.output_key = output_key

    @staticmethod
    def majority_vote(evals):

        evals = [evals[i].model_dump(mode="json") for i in evals]

        metrics = [
            "validity_outcome",
            "intent_outcome",
            "consistency_outcome",
            "disambiguation_outcome",
        ]
        reason_metrics = [
            "validity_reason",
            "intent_reason",
            "consistency_reason",
            "disambiguation_reason",
        ]
        all_metric, reason_metric = {}, {}
        for i in range(len(metrics)):
            all_metric[metrics[i]] = [eval[metrics[i]] for eval in evals]
            reason_metric[metrics[i]] = [eval[reason_metrics[i]] for eval in evals]

        outcome = {}
        feedback = " "
        for i, metric in enumerate(metrics):
            vote_count = Counter(all_metric[metric])
            most_common = vote_count.most_common()

            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                majority = "invalid"  # tie-breaker defaults to 'invalid'
            else:
                majority = most_common[0][0]

            if majority == "invalid":
                temp_feeback = [
                    reason_metric[metric][j]
                    for j in range(len(all_metric[metric]))
                    if all_metric[metric][j] == "invalid"
                ]
                feedback = " ".join(temp_feeback)
            outcome[metric] = {"majority_vote": majority, "feedback": feedback}

        return outcome

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        config = self._get_flow_config(config)
        # sample implementation commnted and empty response being generated
        """
        outcome = ConvResponseAggregator.majority_vote(state.conv_eval_result)
        response = ConversationAggregatorResponse(outcome=outcome)
        """

        response = {}
        return {self.output_key: response}
