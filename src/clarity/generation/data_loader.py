"""Convert benchmark seeds to generation examples; reject out-of-scope data."""

from clarity.generation.task_flows.components.data_types import ConversationExample
from clarity.utils.file_io import load_jsonl


class UtteranceDataLoader:
    def __call__(self, data_path, scenario, dataset_dict, column_dict, values_dict):
        examples = []
        for index, record in enumerate(load_jsonl(data_path), 1):
            db_id = record.get("db_id")
            if record.get("dataset") is not None or db_id not in dataset_dict:
                raise ValueError(f"Seed {index} must name a released Spider/BIRD database")
            sql = record.get("expectedQuery")
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError(f"Seed {index} requires a single parent SQL string")
            examples.append(
                ConversationExample(
                    example_id=index,
                    db_id=db_id,
                    question=record["question"],
                    expected_query=sql,
                    template=record.get("template"),
                    completion=record.get("completion"),
                    synonymsStatus=record.get("synonymsStatus"),
                    initial_state=record.get("initialState"),
                    current_state=record.get("currentState"),
                    history=record.get("history"),
                    type=scenario,
                    spider_schema=dataset_dict[db_id],
                    spider_columns=column_dict[db_id],
                    schema_values=values_dict.get(db_id, {}),
                )
            )
        return examples
