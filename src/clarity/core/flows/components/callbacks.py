from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class LoggingChainPromptInputCallback(BaseCallbackHandler):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_chat_model_start(self, chat_id: UUID, prompts: list[str], **kwargs: Any):
        self.logger.debug("Input prompt:\n%s", prompts[0][-1].content)
