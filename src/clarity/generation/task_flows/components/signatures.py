from pydantic import BaseModel, Field

from clarity.core.flows.components.data_types import NULL


class ParserResponse(BaseModel):
    columns: dict = Field(
        NULL, description="A dictionary of table-columns extracted from SQL query"
    )
    value_pairs: dict = Field(
        NULL, description="A dictionary of table column-value pairs extracted from SQL query"
    )
    selected: list[dict] = Field(
        NULL, description="A list of dictionaries of table-column selected for data generation"
    )


class RetrieverClassificationResponse(BaseModel):
    response: list[dict] = Field(
        NULL,
        description="A list of dictionaries for each column of the given list contain three keys: 'column_name', 'outcome', and 'reason' with column name, valid or invalid assessment and reason behind the outcome values respectively",
    )


class RetrieverResponse(BaseModel):
    group: list[str] = Field(NULL, description="A list containing similar columns")
    coi: str = Field(NULL, description="Column of interest")


class RetrieverSingleColValueResponse(BaseModel):
    group: list[str] = Field(
        NULL, description="A list containing unique values in the column of interest."
    )
    coi: str = Field(NULL, description="Column of interest")
    voi: str = Field(NULL, description="Value of interest")


class RetrieverMultiColValueResponse(BaseModel):
    group: list[str] = Field(
        NULL, description="A list of columns containing the unique value of interest."
    )
    coi: str = Field(NULL, description="Columns of interest")
    voi: str = Field(NULL, description="Value of interest")


class AmbiguityTermResponse(BaseModel):
    group: list[str] = Field(NULL, description="A list containing selected similar columns")
    term: str = Field(NULL, description="An ambiguous term for the selected similar columns")
    reason: str = Field(NULL, description="A reason behind the ambiguous term generation")


class AmbiguityValueTermResponse(BaseModel):
    group: list[str] = Field(NULL, description="A list containing selected similar values")
    term: str = Field(NULL, description="An ambiguous value for the selected similar values")
    reason: str = Field(NULL, description="A reason behind the ambiguous value generation")


class ConfusionTermResponse(BaseModel):
    term: str = Field(NULL, description="A confusing term for the database schema")
    reason: str = Field(NULL, description="The reasoning behind the confusing term generation")


class ConfusionValueResponse(BaseModel):
    term: str = Field(NULL, description="A confusing value for the given column of interest")
    reason: str = Field(NULL, description="The reasoning behind the confusing value generation")


class GeneralResponse(BaseModel):
    outcome: str = Field(
        NULL, description="Outcome for term against the given criterion as 'valid' or 'invalid'"
    )
    reason: str = Field(
        NULL,
        description="A reason behind the decision. The reason should not contain any special characters",
    )


class TermEvaluatorResponse(BaseModel):
    similarity_outcome: str = Field(NULL, description="Outcome for similarity criterion")
    similarity_reason: str = Field(NULL, description="A reason behind the decision")
    dissimilarity_outcome: str = Field(NULL, description="Outcome for dissimilarity criterion")
    dissimilarity_reason: str = Field(NULL, description="A reason behind the decision")
    ac_outcome: str = Field(NULL, description="Outcome for ambiguity/confusion criterion")
    ac_reason: str = Field(NULL, description="A reason behind the decision")


class TermAggregatorResponse(BaseModel):
    similarity_outcome: str = Field(
        NULL, description="Majority vote outcome for similarity criterion"
    )
    similarity_reason: str = Field(NULL, description="A reason behind the decision")
    dissimilarity_outcome: str = Field(
        NULL, description="Majority vote outcome for dissimilarity criterion"
    )
    dissimilarity_reason: str = Field(NULL, description="A reason behind the decision")
    ac_outcome: str = Field(
        NULL, description="Majority vote outcome for ambiguity/confusion criterion"
    )
    ac_reason: str = Field(NULL, description="A reason behind the decision")


class UtteranceGenerationResponse(BaseModel):
    nl_query: str = Field(
        NULL,
        description="A natural language query representation of the given SQL query",
    )
    ac_query: str = Field(
        NULL,
        description="An ambiguous variant of the generated natural language query",
    )


class UtteranceEvaluatorResponse(BaseModel):
    consistency_outcome: str = Field(NULL, description="Outcome for consistency criterion")
    consistency_reason: str = Field(NULL, description="A reason behind the decision")
    ac_outcome: str = Field(NULL, description="Outcome for ambiguity/confusion criterion")
    ac_reason: str = Field(NULL, description="A reason behind the decision")


class UtteranceAggregatorResponse(BaseModel):
    consistency_outcome: str = Field(
        NULL, description="Majority vote outcome for consistency criterion"
    )
    consistency_reason: str = Field(NULL, description="A reason behind the decision")
    ac_outcome: str = Field(
        NULL, description="Majority vote outcome for ambiguity/confusion criterion"
    )
    ac_reason: str = Field(NULL, description="A reason behind the decision")


class ConversationGeneratorResponse(BaseModel):
    conversation: list[dict] = Field(
        [],
        description="a list of dictionaries containing the conversation between agent and user ",
    )


class ConversationEvaluationResponse(BaseModel):
    validity_outcome: str = Field(
        description="valid or invalid response for assessing the ambiguity of the input user request"
    )
    validity_reason: str = Field(description="Reason behind the outcome")
    intent_outcome: str = Field(
        description="valid or invalid response for assessing the intent of the input user request against original request"
    )
    intent_reason: str = Field(description="Reason behind the outcome")
    disambiguation_outcome: str = Field(
        description="valid or invalid response for assessing the correctness of disambigutaion conversation "
    )
    disambiguation_reason: str = Field(description="Reason behind the outcome")
    consistency_outcome: str = Field(
        description="valid or invalid response for assessing the consistency of the agent response against the underlying SQL query"
    )
    consistency_reason: str = Field(description="Reason behind the outcome")


class ConversationAggregatorResponse(BaseModel):
    outcome: dict = Field(NULL, description="Aggregated response of conversation evaluators")


class TCClassificationResponse(BaseModel):
    category: str = Field(
        NULL,
        description="A decision on classifying the question into one of the categories: translatable  or  untranslatable",
    )
    reason: str = Field(
        NULL,
        description="The reason behind the classification. The reason should not contain any special characters",
    )


class TCCheckACResponse(BaseModel):
    terms: list[str] = Field(
        NULL,
        description="A list of terms that refer to a column but do not exactly match any column names in the schema ",
    )
    reason: str = Field(
        NULL,
        description="The reason behind the term(s) identification. The reason should not contain any special characters",
    )


class TCVerifyACResponse(BaseModel):
    outcome: str = Field(NULL, description="'yes' or 'no' response of the assessment")
    reason: str = Field(
        NULL,
        description="The reason behind the outcome. The reason should not contain any special characters",
    )


class ConversationResponse(BaseModel):
    example: dict
    parser_result: dict = None
    retriever_result: dict = None
    retriever_class_result: dict = None
    term_gen_result: dict = None
    term_eval_result: dict = None
    term_agg_result: dict = None
    utt_gen_result: dict = None
    utt_eval_result: dict = None
    utt_agg_result: dict = None
    conv_gen_result: dict = None
    conv_eval_result: dict = None
    conv_agg_result: dict = None
    sample_selector: dict = None
