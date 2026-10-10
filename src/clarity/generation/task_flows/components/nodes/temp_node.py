import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from clarity.generation.task_flows.components.states import ConversationState
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.nodes.base import UtteranceNodeBase
from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class TempNode(UtteranceNodeBase):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
