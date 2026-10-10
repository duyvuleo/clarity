import sys
from enum import Enum

from pydantic import Field, model_validator

from clarity.retrieval.config import RetrievalConfig

from clarity.core.flows.components.data_types import (
    ClassConfig,
    Example,
    PromptConfigKey,
    Task,
)

LOGGER_NAME = sys.intern("conversation-flow")


class ConversationGeneratorTask(Task, extra="allow"):
    dataset: str = Field(pattern="^(spider|bird)$")
    data_root: str = "../data"
    max_examples: int | None = Field(default=None, ge=1)
    multifacet: int = Field(default=1, ge=1, le=3)
    retrieval_type: str = Field(default="llm", pattern="^(llm|hybrid|dense|sparse)$")
    schema_retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    similarity_criteria: str = "lexical"
    llm_retrieval: list[dict] = Field(
        default_factory=list,
        description="Optional model configurations for retrieval or post-filtering.",
    )

    llm_gen_term: dict = Field(description="Configuration for the llm as term generator.")

    llm_gen_utt: dict = Field(description="Configuration for the llm as utterance generator.")

    llm_eval_term: list[dict] = Field(
        description="A list of configuration for the llms as term evaluators."
    )

    llm_eval_utt: list[dict] = Field(
        description="A list of configuration for the llms as term evaluators."
    )

    llm_sample_selectors: list[dict] = Field(
        description="A list of configuration for the generated sample_selectors."
    )

    data_loader: ClassConfig = Field(description="Class config for the data loader.")
    recursion_limit: int = Field(default=10, description="Recursion limit for Langgraph operations")
    ac_type: str = Field(
        default="None",
        description="Type of ambiguity needs to be introduced",
    )

    @model_validator(mode="after")
    def validate_retrieval(self):
        if self.ac_type == "column_ambiguity":
            dense = self.retrieval_type == "dense" or (
                self.retrieval_type == "hybrid" and self.schema_retrieval.alpha > 0
            )
            if dense and self.schema_retrieval.embedding is None:
                raise ValueError("Configure schema_retrieval.embedding for dense/hybrid search")
            if (
                self.retrieval_type == "llm" or self.schema_retrieval.llm_filter
            ) and not self.llm_retrieval:
                raise ValueError("llm_retrieval is required for LLM retrieval or post-filtering")
        return self


class ConversationExample(Example):
    example_id: int
    db_id: str | None = None
    dataset: str | None = None
    template: str | None = None
    completion: str | None = None
    synonymsStatus: str | None = None
    db_schema: str | None = None
    initial_state: str | None = None
    current_state: str | None = None
    question: str | None = None
    expected_query: str | list[str] | None = None
    history: str | None = None
    category: str | None = None
    sql_columns: list[str] | None = None
    new_query: str | None = None
    group: list[list[str]] | None = None
    term: list[str] | str | None = None
    reason: str | None = None
    confidence: list[str] | None = None
    true_columns: str | None = None
    dataset_schemas: dict | None = None
    to_remove: list[dict] | None = None
    type: str | None = None
    cois: list | None = None
    coi: str | list | tuple | None = None
    coi_table: str | None = None
    cois_pair: list | None = None
    voi: str | None = None
    coi_value: list | None = None
    subset_columns: list | None = None
    dataset_columns: list | None = None
    history_term: str | None = None
    history_conv: str | None = None
    ac_query: str | None = None
    conversation: list | None = Field(default_factory=list)
    mtrue_columns: list[str] | None = None
    spider_schema: dict | None = None
    schema_values: dict | None = None
    column_values: list | None = None
    spider_columns: dict | None = None
    cols_to_compare: list | None = None
    vals_to_compare: list | None = None
    ac_group: list | None = None
    iteration: int = 0
    retriever_decision: list[bool] | None = None
    retrieval_hits: dict = Field(default_factory=dict)
    term_decision: list[bool] | None = None


class UtterancePromptConfigKey(PromptConfigKey):
    LEXICAL_TERM = "lexical_term_prompt"
    LEXICAL_AMBIGUITY = "lexical_term_ambiguity_eval_prompt"
    LEXICAL_SIMILARITY = "lexical_term_similarity_eval_prompt"
    LEXICAL_DISSIMILARITY = "lexical_term_dissimilarity_eval_prompt"
    SEMANTIC_TERM = "semantic_term_prompt"
    SEMANTIC_AMBIGUITY = "semantic_term_ambiguity_eval_prompt"
    SEMANTIC_DISSIMILARITY = "semantic_term_dissimilarity_eval_prompt"
    SEMANTIC_SIMILARITY = "semantic_term_similarity_eval_prompt"
    CONFUSION_TERM = "confusion_term_prompt"
    CONFUSION_TERM_VALIDITY = "confusion_validity_term_eval_prompt"
    CONFUSION_DISSIMILARITY = "confusion_term_dissimilarity_eval_prompt"
    CONFUSION_VALUE = "confusion_value_prompt"
    CONFUSION_VALUE_VALIDITY = "confusion_validity_value_eval_prompt"
    UTT_GENERATOR = "utt_generator_prompt"
    UTT_SINGLE_COL_VALUE_AMBIGUITY = "utt_single_col_value_ambiguity_eval_prompt"
    UTT_MULTI_COLS_VALUE_AMBIGUITY = "utt_multi_cols_value_ambiguity_eval_prompt"
    UTT_GENERATOR_VALUE = "utt_value_generator_prompt"
    UTT_CONSISTENCY_AMBIGUITY = "utt_ambiguity_consistency_eval_prompt"
    UTT_CONSISTENCY_CONFUSION = "utt_confusion_consistency_eval_prompt"
    UTT_CONFUSION_COLUMN = "utt_confusion_column_eval_prompt"
    UTT_CONFUSION_VALUE = "utt_confusion_value_eval_prompt"
    UTT_SINGLE_COL_VALUE_AMBIGUITY_GENERATOR = "utt_single_col_value_ambiguity_generator_prompt"
    UTT_MULTI_COLS_VALUE_AMBIGUITY_GENERATOR = "utt_multi_cols_value_ambiguity_generator_prompt"
    UTT_CONSISTENCY_AMB = "utt_ambiguity_consistency_eval_prompt"
    UTT_LEXICAL_AMBIGUITY = "utt_lexical_ambiguity_eval_prompt"
    UTT_SEMANTIC_AMBIGUITY = "utt_semantic_ambiguity_eval_prompt"
    UTT_CONFUSION = "utt_confusion_eval_prompt"
    CONV_GENERATOR_SINGLE = "conv_generator_prompt_single"
    CONV_GENERATOR_MULTI = "conv_generator_prompt_multi"
    CONV_GENERATOR_CONFUSION_COLUMN = "conv_generator_prompt_confusion_column"
    CONV_GENERATOR_CONFUSION_VALUE = "conv_generator_prompt_confusion_value"
    CONV_EVALUATOR = "conv_evaluator_prompt"
    LLM_RETRIEVAL = "llm_retrieval_prompt"
    SINGLE_COL_VALUE_AMBIGUITY = "single_col_value_ambiguity_prompt"
    MULTI_COLS_VALUE_AMBIGUITY = "multi_cols_value_ambiguity_prompt"
    SINGLE_COL_VALUE_TERM_SIMILARITY_EVAL_PROMPT = "single_col_value_term_similarity_eval_prompt"
    MULTI_COLS_VALUE_TERM_SIMILARITY_EVAL_PROMPT = "multi_cols_value_term_similarity_eval_prompt"
    HYBRID_LEXICAL_RETRIEVAL = "hybrid_lexical_retrieval_prompt"
    HYBRID_SEMANTIC_RETRIEVAL = "hybrid_semantic_retrieval_prompt"
    TC1 = "test_case_classify"
    TC2_1 = "test_case_ac"
    TC2_2 = "verify_ac"


class DatasetKey(str, Enum):
    ERROR = "error"
    CATEGORY = "category"
    INTIAL_STATE = "initialState"
    CURRENT_STATE = "currentState"
    DB_ID = "db_id"


class SettingKey(str, Enum):
    COLUMN_AMBIGUITY = "column_ambiguity"
    COLUMN_CONFUSION = "column_confusion"
    VALUE_CONFUSION = "value_confusion"
    LLM_RETRIEVAL = "llm"
    HYBRID_RETRIEVAL = "hybrid"
    DENSE_RETRIEVAL = "dense"
    LEXICAL_SIMILARITY = "lexical"
    SEMANTIC_SIMILARITY = "semantic"
    VALUE_AMBIGUITY_MULTI_COLS = "multi_cols_value_ambiguity"
    VALUE_AMBIGUITY_SIGL_COL = "single_col_value_ambiguity"
    VALUE_AMBIGUITIES = [VALUE_AMBIGUITY_SIGL_COL, VALUE_AMBIGUITY_MULTI_COLS]
    EXACT_MATCHES = "exact_matches"
    RANGE_MATCHES = "range_matches"
