"""
Synthetic Conversational AC Data Generator
- A pipeline that consists of a series of generators and evaluators.
- Different modules are used to generate the conversation step by step, consisting of
    - ac term generator and evaluator
    - ac utterance generator and evaluator
    - conversation generator and evaluator
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from clarity.generation.task_flows.components.data_types import (
    LOGGER_NAME,
    UtterancePromptConfigKey,
    SettingKey,
)
from clarity.generation.task_flows.components.edges.conv_evaluator_decision import (
    ConvEvaluatorDecision,
)
from clarity.generation.task_flows.components.edges.term_evaluator_decision import (
    TermEvaluatorDecision,
)
from clarity.generation.task_flows.components.edges.retriever_decision import RetrieverDecision
from clarity.generation.task_flows.components.edges.parser_decision import ParserDecision
from clarity.generation.task_flows.components.edges.term_decision import ACTermDecision
from clarity.generation.task_flows.components.edges.utt_evaluator_decision import (
    UtteranceEvaluatorDecision,
)
from clarity.generation.task_flows.components.nodes.ac_conv_evaluator import ACConversationEvaluator
from clarity.generation.task_flows.components.nodes.ac_conv_gen import ACConversationGenerator
from clarity.generation.task_flows.components.nodes.ac_term_evaluator import ACTermEvaluator
from clarity.generation.task_flows.components.nodes.ac_term_gen import ACTermGenerator
from clarity.generation.task_flows.components.nodes.ac_utt_evaluator import ACUtteranceEvaluator
from clarity.generation.task_flows.components.nodes.ac_utt_gen import ACUtteranceGenerator
from clarity.generation.task_flows.components.nodes.conv_aggregator import ConvResponseAggregator
from clarity.generation.task_flows.components.nodes.conversation_writer import ResponseGenerator
from clarity.generation.task_flows.components.nodes.retreiver import SchemaRetriever
from clarity.generation.task_flows.components.nodes.parser import SQLParser
from clarity.generation.task_flows.components.nodes.sample_selector import SampleSelector
from clarity.generation.task_flows.components.nodes.simple import ConversationFlowSetup
from clarity.generation.task_flows.components.nodes.term_aggregator import TermResponseAggregator
from clarity.generation.task_flows.components.nodes.utterance_aggregator import (
    UtteranceResponseAggregator,
)
from clarity.generation.task_flows.components.signatures import *
from clarity.generation.task_flows.components.states import (
    ConversationState,
    ConvEvaluatorState,
    ConvOutputState,
    InputState,
)
from clarity.generation.task_flows.flows.static_flow import StaticFlow

logger = logging.getLogger(LOGGER_NAME)


class ConvUtteranceGenEvalsV1(StaticFlow):
    def create_ac_term_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            (
                "ac_term_generator",
                AmbiguityTermResponse,
            ),
        ]

        # Add nodes
        for node_name, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                ACTermGenerator(output_pydantic_object=output_pydantic_object),
            )

        # Add edges
        for node_name, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_term_evaluator_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            (
                "ac_term_evaluator",
                GeneralResponse,
            ),
        ]

        # Add nodes
        for node_name, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                ACTermEvaluator(
                    output_pydantic_object=output_pydantic_object,
                ),
            )

        # Add edges
        for node_name, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_utt_evaluator_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            (
                "ac_utt_evaluator",
                GeneralResponse,
            ),
        ]

        # Add nodes
        for node_name, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                ACUtteranceEvaluator(
                    output_pydantic_object=output_pydantic_object,
                ),
            )

        # Add edges
        for (
            node_name,
            _,
        ) in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_utt_generator_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            (
                "seed_utterance_generator",
                UtterancePromptConfigKey.UTT_GENERATOR,
                UtteranceGenerationResponse,
            ),
        ]

        # Add nodes
        for node_name, prompt_key, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                ACUtteranceGenerator(
                    prompt_key=prompt_key, output_pydantic_object=output_pydantic_object
                ),
            )

        # Add edges
        for node_name, _, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_term_aggregator_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        graph.add_node("term_response_aggregator", TermResponseAggregator())

        graph.add_edge(START, "term_response_aggregator")
        graph.add_edge("term_response_aggregator", END)
        agent = graph.compile()

        return agent

    def create_utt_aggregator_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        graph.add_node("utt_response_aggregator", UtteranceResponseAggregator())

        graph.add_edge(START, "utt_response_aggregator")
        graph.add_edge("utt_response_aggregator", END)
        agent = graph.compile()

        return agent

    def create_conv_aggregator_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        graph.add_node("conv_response_aggregator", ConvResponseAggregator())

        graph.add_edge(START, "conv_response_aggregator")
        graph.add_edge("conv_response_aggregator", END)
        agent = graph.compile()

        return agent

    def create_conv_generator_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            (
                "conversation_generator",
                ConversationGeneratorResponse,
            ),
        ]

        # Add nodes
        for node_name, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                ACConversationGenerator(output_pydantic_object=output_pydantic_object),
            )

        # Add edges
        for node_name, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_schema_retriever(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            ("schema_retriever", RetrieverResponse),
        ]

        # Add nodes
        for node_name, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                SchemaRetriever(output_pydantic_object=output_pydantic_object),
            )

        # Add edges
        for node_name, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_schema_value_retriever(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            ("schema_retriever", RetrieverSingleColValueResponse),
        ]

        # Add nodes
        for node_name, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                SchemaRetriever(output_pydantic_object=output_pydantic_object),
            )

        # Add edges
        for node_name, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_parser(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [
            ("sql_parser", ParserResponse),
        ]

        # Add nodes
        for node_name, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                SQLParser(output_pydantic_object=output_pydantic_object),
            )

        # Add edges
        for node_name, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_conv_evaluator_agent(self, judge) -> CompiledStateGraph:
        graph = StateGraph(ConvEvaluatorState)

        predictors = [
            (
                "ac_conv_evaluator",
                UtterancePromptConfigKey.CONV_EVALUATOR,
                ConversationEvaluationResponse,
            ),
        ]

        # Add nodes
        for node_name, prompt_key, output_pydantic_object in predictors:
            graph.add_node(
                node_name,
                ACConversationEvaluator(
                    prompt_key=prompt_key,
                    output_pydantic_object=output_pydantic_object,
                    judge=judge,
                ),
            )

        # Add edges
        for node_name, _, _ in predictors:
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create_conv_response_generation_agent(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        graph.add_node("response_generator", ResponseGenerator())

        graph.add_edge(START, "response_generator")
        graph.add_edge("response_generator", END)
        agent = graph.compile()

        return agent

    def create_sample_selector(self) -> CompiledStateGraph:
        graph = StateGraph(ConversationState)

        predictors = [("select_sample")]

        # Add nodes and edges
        for node_name in predictors:
            graph.add_node(node_name, SampleSelector())
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        agent = graph.compile()
        return agent

    def create(self, ac_type: SettingKey) -> CompiledStateGraph:
        parser_agent = self.create_parser()
        if ac_type in [SettingKey.COLUMN_AMBIGUITY]:
            retriever_agent = self.create_schema_retriever()
        elif ac_type in [
            SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
            SettingKey.VALUE_AMBIGUITY_SIGL_COL,
        ]:
            retriever_agent = self.create_schema_value_retriever()
        ac_term_agent = self.create_ac_term_agent()
        eval_term_agent = self.create_term_evaluator_agent()
        term_aggregator_agent = self.create_term_aggregator_agent()
        utt_generator_agent = self.create_utt_generator_agent()
        utt_evaluator_agent = self.create_utt_evaluator_agent()
        utt_aggregator_agent = self.create_utt_aggregator_agent()
        conv_generator_agent = self.create_conv_generator_agent()
        eval_conv_agent1 = self.create_conv_evaluator_agent(judge=0)
        eval_conv_agent2 = self.create_conv_evaluator_agent(judge=1)
        conv_aggregator_agent = self.create_conv_aggregator_agent()
        sample_selector = self.create_sample_selector()

        response_generator_agent = self.create_conv_response_generation_agent()

        builder = StateGraph(ConversationState, input=InputState, output=ConvOutputState)

        # Add nodes
        builder.add_node("setup", ConversationFlowSetup())
        builder.add_node("parser_agent", parser_agent)
        if ac_type in [
            SettingKey.COLUMN_AMBIGUITY,
            SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
            SettingKey.VALUE_AMBIGUITY_SIGL_COL,
        ]:
            builder.add_node("retriever_agent", retriever_agent)
        builder.add_node("ac_term_agent", ac_term_agent)
        builder.add_node("term_eval", eval_term_agent)
        builder.add_node("term_aggregator", term_aggregator_agent)
        builder.add_node("utterance_generator", utt_generator_agent)
        builder.add_node("utterance_evaluator", utt_evaluator_agent)
        builder.add_node("utterance_aggregator", utt_aggregator_agent)
        builder.add_node("conversation_generator", conv_generator_agent)
        builder.add_node("conversation_evaluator1", eval_conv_agent1)
        builder.add_node("conversation_evaluator2", eval_conv_agent2)
        builder.add_node("conv_aggregator", conv_aggregator_agent)
        builder.add_node("select_sample", sample_selector)

        builder.add_node("response_generator", response_generator_agent)

        # Add edges
        builder.add_edge(START, "setup")
        builder.add_edge("setup", "parser_agent")

        if ac_type in [
            SettingKey.COLUMN_AMBIGUITY,
            SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
            SettingKey.VALUE_AMBIGUITY_SIGL_COL,
        ]:
            builder.add_conditional_edges(
                "parser_agent",
                ParserDecision(),
                {True: "retriever_agent", False: "response_generator"},
            )

            builder.add_conditional_edges(
                "retriever_agent",
                RetrieverDecision(),
                {True: "ac_term_agent", False: "parser_agent"},
            )

        else:
            builder.add_conditional_edges(
                "parser_agent",
                ParserDecision(),
                {True: "ac_term_agent", False: "response_generator"},
            )

        builder.add_conditional_edges(
            "ac_term_agent",
            ACTermDecision(),
            {True: "term_eval", False: "parser_agent"},
        )

        builder.add_edge("term_eval", "term_aggregator")

        builder.add_conditional_edges(
            "term_aggregator",
            TermEvaluatorDecision(),
            {True: "utterance_generator", False: "ac_term_agent"},
        )

        builder.add_edge("utterance_generator", "utterance_evaluator")
        builder.add_edge("utterance_evaluator", "utterance_aggregator")
        builder.add_conditional_edges(
            "utterance_aggregator",
            UtteranceEvaluatorDecision(),
            {True: "conversation_generator", False: "utterance_generator"},
        )
        builder.add_edge("conversation_generator", "select_sample")
        builder.add_edge("select_sample", "response_generator")

        ###place holders for conversation evaluators - if needed
        # builder.add_edge("conversation_generator", "conversation_evaluator1")
        # builder.add_edge("conversation_generator", "conversation_evaluator2")
        # builder.add_edge("conversation_evaluator1", "conv_aggregator")
        # builder.add_edge("conversation_evaluator2", "conv_aggregator")

        # builder.add_conditional_edges(
        #    "conv_aggregator",
        #    ConvEvaluatorDecision(),
        #    {True: "response_generator", False: "utterance_generator"},
        # )

        builder.add_edge("response_generator", END)

        flow = builder.compile()
        return flow
