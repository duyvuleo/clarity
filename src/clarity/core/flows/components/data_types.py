import sys
from enum import Enum

from pydantic import BaseModel, Field

NULL = sys.intern("null")
ERROR_MESSAGE_TAG = "[ERROR]:"


class Example(BaseModel):
    pass


class EvalResult(BaseModel):
    is_match: int
    gold: str | None = None
    pred: str | None = None


class PromptConfigKey:
    PROMPTS = "prompts"

    @classmethod
    def contains(cls, value: str) -> bool:
        return value in cls.__dict__.values()


class FlowConfigKey(str, Enum):
    THREAD_ID = "thread_id"
    CONFIGURABLE = "configurable"
    RECURSION_LIMIT = "recursion_limit"
    LLM = "llm"


class ClassConfig(BaseModel):
    name: str = Field(description="Path to the Python file and class name")
    args: dict | None = Field(default=None, description="Arguments to be passed to the class")


class Task(BaseModel):
    """Represents a specific task for APO"""

    name: str = Field(description="Name of the task.")
    description: str | None = Field(default=None, description="Description of the task.")
    train_data_path: str | None = Field(default=None, description="Path to the training data.")
    dev_data_path: str | None = Field(default=None, description="Path to the development data.")
    prompt_config_path: str = Field(description="Path to the prompt config file.")
