import logging

from langchain_core.runnables import RunnableConfig
from clarity.generation.task_flows.components.data_types import LOGGER_NAME
from clarity.generation.task_flows.components.states import ConversationState

from clarity.core.flows.components.nodes.simple import Setup
from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class ConversationFlowSetup(Setup):
    def __call__(self, state: ConversationState, config: RunnableConfig) -> dict:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        return {}
