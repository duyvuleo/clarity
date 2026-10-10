import random

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from clarity.core.flows.components.nodes.base import FlowNodeBase


class Setup(FlowNodeBase):
    def __call__(self, state: BaseModel, config: RunnableConfig) -> dict:
        return {}


class BatchSampler(FlowNodeBase):
    def __call__(self, state: BaseModel, config: RunnableConfig) -> dict:
        flow_config = self._get_flow_config(config)
        train_data = flow_config.train_data
        batch_size = (
            flow_config.optimizer.batch_size
            if hasattr(flow_config, "optimizer")
            else flow_config.batch_size
        )
        batch_examples = random.sample(train_data, batch_size)
        return {"batch_examples": batch_examples}
