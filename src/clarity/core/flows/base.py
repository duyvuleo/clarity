import importlib.util
from abc import ABC

from langgraph.graph.state import CompiledStateGraph

from clarity.core.flows.components.data_types import ClassConfig, Task


class BaseFlow(ABC):
    def __init__(self, task: Task, flow_png_path: str | None = None) -> None:
        """
        :param task: path to the task config file
        :param flow_png_path: path to save the flow graph png
        """
        self._task = task
        self.flow_png_path = flow_png_path
        self.flow: CompiledStateGraph = None

    @staticmethod
    def _load_class_from_module_path(class_config: ClassConfig) -> object | None:
        if not class_config:
            return None
        module_path, class_name = class_config.name.split(":")
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target_class = getattr(module, class_name)
        return target_class(**class_config.args) if class_config.args else target_class()

    def prep_task(self) -> Task: ...

    def viz_flow(self) -> None:
        from langchain_core.runnables.graph import MermaidDrawMethod

        self.flow.get_graph(xray=True).draw_mermaid_png(
            draw_method=MermaidDrawMethod.API, output_file_path=self.flow_png_path
        )

    def print_flow(self) -> None: ...

    def create(self) -> CompiledStateGraph: ...

    async def arun(self) -> None: ...

    def run(self) -> str: ...
