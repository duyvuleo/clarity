from typing import Annotated, Any, Dict

from langgraph.managed.is_last_step import RemainingSteps
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import ConversationExample
from clarity.generation.task_flows.components.signatures import *


def update_dict(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    current.update(update)
    return current


def equals_examples(
    current: ConversationExample,
    update: ConversationExample,
) -> ConversationExample:
    assert current == update
    return update


def update_nested_dict(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if key in current and isinstance(current[key], dict) and isinstance(value, dict):
            # Recursively update nested dictionaries
            current[key] = update_dict(current[key], value)
        else:
            # Overwrite or add new key
            current[key] = value
    return current


class InputState(BaseModel):
    example: Annotated[ConversationExample, equals_examples]


class ConvOutputState(BaseModel):
    response: ConversationResponse = None


class ParserState(InputState):
    """
    The local state of the Parser flow
    Attributes:
        columns: A list of columns extracted from the SQL query
        values: A list of column-value pairs extracted from the SQL query
        selection: A list of columns selected for the data generation
    """

    parser_result: Annotated[dict, update_dict]


class RetrieverClassificationState(InputState):
    """
    The local state of the Retriever flow
    Attributes:
        response:  A list of dictionaries for each column of the given list contain three keys: 'column_name', 'outcome', and 'reason'")

    """

    retriever_class_result: Annotated[dict, update_dict]


class RetrieverState(InputState):
    """
    The local state of the Retriever flow
    Attributes:
        group: A dictionary containing the similar columns
    """

    retriever_result: Annotated[dict, update_dict]


class AmbiguityTermState(InputState):
    """
    The local state of the Ambiguity Tern Generator flow
    Attributes:
        group: A list containing lists of similar columns
        reason: A reason behind the grouping decision
        ambiguous_entity: A list containing ambigious terms for the columns
    """

    term_gen_result: Annotated[dict, update_dict]


class TermEvaluatorState(InputState):
    """
    The local state of the Term Evaluator flow
    Attributes:
        outcome: Outcome for each ambiguous term as 'valid' or 'invalid'
        reason: The reason behind the decision
    """

    term_eval_result: Annotated[dict, update_nested_dict]


class TermAggregationState(InputState):
    """
    The local state to aggregate the results of multiple term evaluators
    Attributes:
        outcome: The binary response whether the response is correct or not.
        reason: Recording reason if the ouctome is incorrect
    """

    term_agg_result: Annotated[dict, update_dict]
    remaining_steps: RemainingSteps


class UtteranceGenerationState(InputState):
    """
    The local state of the Utterance Generator flow

    """

    utt_gen_result: Annotated[dict, update_dict]


class UtteranceEvaluatorState(InputState):
    """
    The local state of the Utterance Evalauator flow

    """

    utt_eval_result: Annotated[dict, update_dict]


class UtteranceAggregationState(InputState):
    """
    The local state to aggregate the results of multiple utterance evaluators
    """

    utt_agg_result: Annotated[dict, update_dict]
    remaining_steps: RemainingSteps


class ConvGenerationState(InputState):
    """
    The local state of the Conversation Generator flow

    """

    conv_gen_result: ConversationGeneratorResponse = None


class ConvEvaluatorState(InputState):
    """
    The local state of the Conversation Evaluator flow

    """

    conv_eval_result: Annotated[dict, update_dict]


class ConvAggregationState(InputState):
    """
    The local state of the Conversation Generator flow

    """

    conv_agg_result: ConversationAggregatorResponse = None
    remaining_steps: RemainingSteps


class SampleSelectorState(InputState):
    """
    The local state of the Sample Selector

    """

    sample_selector: dict = None


class ConversationState(
    ParserState,
    RetrieverState,
    RetrieverClassificationState,
    AmbiguityTermState,
    TermEvaluatorState,
    TermAggregationState,
    UtteranceGenerationState,
    UtteranceEvaluatorState,
    UtteranceAggregationState,
    ConvGenerationState,
    ConvEvaluatorState,
    ConvAggregationState,
    ConvOutputState,
    SampleSelectorState,
):
    """
    The global state of the Utterance Generation flow.
    """

    pass
