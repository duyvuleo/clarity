from pathlib import Path

from clarity.retrieval.embeddings import create_embedding_provider
from clarity.retrieval.index import SchemaIndex
from clarity.utils.file_io import load_json


def build_schema_index(config, mode, schemas, columns, base_directory):
    options = config.model_copy(deep=True)
    base = Path(base_directory)
    if options.cache_dir:
        options.cache_dir = str((base / options.cache_dir).resolve())
    if options.embedding and options.embedding.cache_folder:
        options.embedding.cache_folder = str((base / options.embedding.cache_folder).resolve())
    needs_dense = mode == "dense" or (mode == "hybrid" and options.alpha > 0)
    if needs_dense and options.embedding is None:
        raise ValueError("Configure schema_retrieval.embedding for dense or hybrid retrieval")
    embedding = create_embedding_provider(options.embedding) if needs_dense else None
    return SchemaIndex(schemas, columns, options, embedding, mode)


def index_from_task_file(path):
    from clarity.generation.task_flows.components.data_types import ConversationGeneratorTask

    path = Path(path).resolve()
    task = ConversationGeneratorTask.model_validate(load_json(path))
    if task.retrieval_type == "llm":
        raise ValueError("Choose retrieval_type hybrid, dense or sparse for conventional search")
    data = (path.parent / task.data_root).resolve() / task.dataset
    return build_schema_index(
        task.schema_retrieval,
        task.retrieval_type,
        load_json(data / "schemas.json"),
        load_json(data / "columns.json"),
        path.parent,
    )
