import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.signatures import ConversationResponse
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.base import FlowNodeBase
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ResponseGenerator(FlowNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))

        config = self._get_flow_config(config)
        state.example.schema_values = None
        response = ConversationResponse(example=state.example.model_dump(mode="json"))

        if state.parser_result:
            response.parser_result = state.parser_result

        if state.retriever_result:
            response.retriever_result = {
                i: state.retriever_result[i].model_dump(mode="json") for i in state.retriever_result
            }

        if state.retriever_class_result:
            # response.retriever_class_result = {
            #    i: state.retriever_class_result[i].model_dump(mode="json")
            #    for i in state.retriever_class_result
            # }

            response.retriever_class_result = {
                i: {
                    j: state.retriever_class_result[i][j].model_dump(mode="json")
                    for j in state.retriever_class_result[i]
                }
                for i in state.retriever_class_result
            }

        if state.term_gen_result:
            response.term_gen_result = {
                i: state.term_gen_result[i].model_dump(mode="json") for i in state.term_gen_result
            }

        if state.term_eval_result:
            response.term_eval_result = {
                i: {
                    j: state.term_eval_result[i][j].model_dump(mode="json")
                    for j in state.term_eval_result[i]
                }
                for i in state.term_eval_result
            }

        if state.term_agg_result:
            response.term_agg_result = {
                i: state.term_agg_result[i].model_dump(mode="json") for i in state.term_agg_result
            }

        if state.utt_gen_result:
            response.utt_gen_result = state.utt_gen_result

        if state.utt_eval_result:
            response.utt_eval_result = {
                i: state.utt_eval_result[i].model_dump(mode="json") for i in state.utt_eval_result
            }

        if state.utt_agg_result:
            response.utt_agg_result = state.utt_agg_result

        if state.conv_gen_result:
            response.conv_gen_result = state.conv_gen_result.model_dump(mode="json")

        if state.sample_selector:
            response.sample_selector = state.sample_selector

        if state.conv_eval_result:
            response.conv_eval_result = {
                i: state.conv_eval_result[i].model_dump(mode="json") for i in state.conv_eval_result
            }

        if state.conv_agg_result:
            response.conv_agg_result = state.conv_agg_result.model_dump(mode="json")

        return {"response": response}
