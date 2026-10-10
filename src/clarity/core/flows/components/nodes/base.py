from abc import ABC, abstractmethod

from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from clarity.core.flows.components.data_types import FlowConfigKey, PromptConfigKey


class FlowNodeBase(ABC):
    def __init__(self, prompt_config_key_cls: PromptConfigKey = PromptConfigKey):
        self.prompt_config_key_cls = prompt_config_key_cls

    @staticmethod
    def _get_flow_config(config: RunnableConfig) -> dict:
        return config[FlowConfigKey.CONFIGURABLE]

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

    def _create_messages(
        self,
        messages: list[AnyMessage],
        role: str,
        meta_prompts: dict,
        output_parser: BaseOutputParser,
        prompt: str,
    ) -> ChatPromptTemplate:
        """
        Prepare input message list for Chat Models
        """
        prompt_template = self._get_prompt_template(meta_prompts, prompt)
        messages.append((role, prompt_template))
        template = ChatPromptTemplate(messages)
        if isinstance(output_parser, (JsonOutputParser, PydanticOutputParser)):
            return template.partial(format_instructions=output_parser.get_format_instructions())
        return template

    @abstractmethod
    def __call__(self, state: BaseModel, config: RunnableConfig) -> dict | bool: ...
