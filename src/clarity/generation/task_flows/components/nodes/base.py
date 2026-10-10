import logging
from abc import ABC
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import (
    LOGGER_NAME,
    ConversationExample,
    UtterancePromptConfigKey,
)

from clarity.models import Llm
from clarity.core.flows.components.callbacks import LoggingChainPromptInputCallback
from langchain_core.callbacks import BaseCallbackHandler

from clarity.core.flows.components.data_types import (
    ERROR_MESSAGE_TAG,
    FlowConfigKey,
    Task,
)


class CapturePromptCallback(BaseCallbackHandler):
    def __init__(self):
        self.prompt_text = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Capture the text(s) that go to the LLM
        # self.prompt_text = prompts
        # print("=== Prompt Sent to LLM ===")
        # print(prompts[0])
        pass


logger = logging.getLogger(LOGGER_NAME)


class UtteranceNodeBase(ABC):
    def __init__(
        self,
        prompt_config_key_cls=UtterancePromptConfigKey,
        prompt_key: str | None = None,
        output_pydantic_object: BaseModel | None = None,
        input_fields: list[str] | None = None,
        output_key: str | None = None,
    ):

        self.prompt_config_key_cls = prompt_config_key_cls
        self.output_pydantic_object = output_pydantic_object
        self.prompt_key = prompt_key
        self.input_fields = input_fields
        self.output_key = output_key

    @staticmethod
    def _get_flow_config(config: RunnableConfig) -> Task:
        return config[FlowConfigKey.CONFIGURABLE][FlowConfigKey.LLM]

    def _get_prompt_template(self, meta_prompts: dict, prompt: str) -> str:
        return meta_prompts[prompt] if self.prompt_config_key_cls.contains(prompt) else prompt

    def _create_prompt(
        self, meta_prompts: dict, output_parser: BaseOutputParser, prompt: str
    ) -> ChatPromptTemplate:
        prompt_template = self._get_prompt_template(meta_prompts, prompt)
        if isinstance(output_parser, (JsonOutputParser, PydanticOutputParser)):
            return ChatPromptTemplate.from_template(prompt_template).partial(
                format_instructions=output_parser.get_format_instructions()
            )
        return ChatPromptTemplate.from_template(prompt_template)

    def _generate(
        self,
        example: ConversationExample,
        llm: Llm,
        meta_prompts: dict,
        output_pydantic_object: BaseModel,
        prompt_key: str,
        input_fields: list[str],
    ) -> Any:

        output_parser = PydanticOutputParser(pydantic_object=output_pydantic_object)
        # output_fixing_parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser, max_retries=10)
        prompt = self._create_prompt(
            meta_prompts=meta_prompts,
            output_parser=output_parser,
            prompt=prompt_key,
        )
        chain = prompt | llm | output_parser  # output_fixing_parser
        capture_callback = CapturePromptCallback()

        try:
            response = chain.invoke(
                dict(
                    [(input_field, getattr(example, input_field)) for input_field in input_fields]
                ),
                config={
                    "callbacks": [capture_callback, LoggingChainPromptInputCallback(logger=logger)]
                },
            )

        except OutputParserException as e:
            error_msg = "{}{}".format(ERROR_MESSAGE_TAG, e)
            logger.warning(error_msg)
            response = error_msg

        return response
