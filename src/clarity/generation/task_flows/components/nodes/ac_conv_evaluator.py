import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.nodes.base import UtteranceNodeBase
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ACConversationEvaluator(UtteranceNodeBase):
    def __init__(
        self,
        prompt_key: str,
        output_pydantic_object: BaseModel,
        input_fields: list[str] = [
            "db_schema",
            "new_query",
            "expected_query",
            "conversation",
        ],
        output_key: str = "conv_eval_result",
        judge: int = 0,
    ):
        super().__init__(
            prompt_key=prompt_key,
            output_pydantic_object=output_pydantic_object,
            input_fields=input_fields,
            output_key=output_key,
        )
        self.judge = judge

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)

        # sample implementation commnted and empty dictionary being returned
        output = {}
        """
        response = self._generate(
            example=state.example,
            llm=flow_config.llm_eval[self.judge],
            meta_prompts=flow_config.meta_prompts,
            output_pydantic_object=self.output_pydantic_object,
            prompt_key=self.prompt_key,
            input_fields=self.input_fields,
        )

        if isinstance(response, str) and ERROR_MESSAGE_TAG in response:

            eval_res = self.output_pydantic_object()
            eval_res.validity_reason = response
            response = eval_res

        output = {self.judge: response}
        """
        return {self.output_key: output}
