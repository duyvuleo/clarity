import logging
from typing import Any
import re
import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import (
    LOGGER_NAME,
    SettingKey,
    UtterancePromptConfigKey,
)
from clarity.generation.task_flows.components.nodes.base import UtteranceNodeBase
from clarity.generation.task_flows.components.signatures import (
    RetrieverResponse,
    RetrieverSingleColValueResponse,
    RetrieverMultiColValueResponse,
    RetrieverClassificationResponse,
)
from clarity.generation.task_flows.components.states import ConversationState

from clarity.utils.terminal import Color
from clarity.core.flows.components.data_types import ERROR_MESSAGE_TAG

logger = logging.getLogger(LOGGER_NAME)


class SchemaRetriever(UtteranceNodeBase):
    def __init__(
        self,
        output_pydantic_object: BaseModel,
        input_fields: list[str] = ["coi", "db_schema", "subset_columns", "category"],
        output_key: str = "retriever_result",
    ):
        self.output_pydantic_object = output_pydantic_object
        self.clss_output_key = "retriever_class_result"
        super().__init__(
            output_pydantic_object=output_pydantic_object,
            input_fields=input_fields,
            output_key=output_key,
        )

    @staticmethod
    def extract_columns_from_dense(data):
        columns = [data[i]["properties"]["column_name"].strip('"') for i in range(len(data))]
        scores = [data[i]["hybrid_score"] for i in range(len(data))]
        return columns, scores

    @staticmethod
    def extract_valid_columns(outcome):
        columns = []
        for element in outcome:
            try:
                if element["outcome"].lower() == "valid":
                    columns.append(element["name"])
            except:
                continue
        return columns

    @staticmethod
    def process_dataset_name(text):
        text = "financialsexmemployeeexpe" if text == "Financials - EXM Employee Expenses" else text
        return re.sub(r"[-_ ]", "", text).lower()

    @staticmethod
    def compute_threshold(n):
        if n <= 100:
            thresh = n * 0.40
        elif n > 100 and n <= 150:
            thresh = n * 0.35
        elif n > 150 and n <= 200:
            thresh = n * 0.25
        elif n > 200 and n <= 250:
            thresh = n * 0.2
        elif n > 250 and n <= 350:
            thresh = n * 0.15
        else:
            thresh = n * 0.10
        return thresh

    def LLM_call(self, state, flow_config, llm):
        response = self._generate(
            example=state.example,
            llm=llm,
            meta_prompts=flow_config.meta_prompts,
            output_pydantic_object=self.output_pydantic_object,
            prompt_key=self.prompt_key,
            input_fields=self.input_fields,
        )

        if isinstance(response, str) and ERROR_MESSAGE_TAG in response:
            eval_res = self.output_pydantic_object()
            response = eval_res
        return response

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))

        flow_config = self._get_flow_config(config)
        state.example.category = flow_config.similarity_criteria
        llm_filter_response = {}
        response = {}

        if state.parser_result["selected"]:
            for m in range(flow_config.multifacet):
                if (
                    SettingKey.COLUMN_AMBIGUITY == flow_config.ac_type
                    or SettingKey.COLUMN_CONFUSION == flow_config.ac_type
                ):
                    table = next(iter(state.parser_result["selected"][m]))
                    state.example.coi = state.example.cois[m]
                    state.example.coi_table = table
                    state.example.db_schema = state.example.spider_schema[table]
                    state.example.dataset_columns = state.example.spider_columns[table]
                    if (
                        flow_config.retrieval_type == SettingKey.DENSE_RETRIEVAL
                        or flow_config.retrieval_type == SettingKey.HYBRID_RETRIEVAL
                    ):
                        key = next(iter(state.parser_result["selected"][m]))

                        # this string processing is linked with retriever - dataset name mapping to generated knowlededge base
                        if state.example.db_id is not None:
                            dataset = state.example.db_id + key
                            state.example.dataset = dataset
                            threshold = round(
                                SchemaRetriever.compute_threshold(
                                    len(state.example.spider_columns[key])
                                )
                            )
                            ### 1-3 are included to include all the columns - this should be removed if retriver is use
                            subset_columns = state.example.spider_columns[key]  # 1
                            subset_columns = [item.lower() for item in subset_columns]  # 2
                            state.example.subset_columns = subset_columns  # 3

                        else:
                            dataset = state.example.dataset
                            threshold = round(
                                SchemaRetriever.compute_threshold(
                                    len(state.example.dataset_columns)
                                )
                            )
                            ### 1-5 below are applicable to spider set as well, but for now using all the columns
                            dataset = SchemaRetriever.process_dataset_name(dataset)  # 1
                            state.example.coi = state.example.cois[m]  # 2
                            subset_columns = flow_config.db.search(
                                state.example.coi, dataset, top_k=threshold
                            )  # 3
                            subset_columns, scores = SchemaRetriever.extract_columns_from_dense(
                                subset_columns
                            )  # 4
                            state.example.subset_columns = subset_columns  # 5

                    if (
                        flow_config.retrieval_type == SettingKey.LLM_RETRIEVAL
                        or flow_config.retrieval_type == SettingKey.HYBRID_RETRIEVAL
                    ):
                        if flow_config.retrieval_type == SettingKey.LLM_RETRIEVAL:
                            self.prompt_key = UtterancePromptConfigKey.LLM_RETRIEVAL
                        else:
                            self.prompt_key = (
                                UtterancePromptConfigKey.HYBRID_SEMANTIC_RETRIEVAL
                                if flow_config.similarity_criteria == SettingKey.SEMANTIC_SIMILARITY
                                else UtterancePromptConfigKey.HYBRID_LEXICAL_RETRIEVAL
                            )
                            self.output_pydantic_object = RetrieverClassificationResponse
                        ### this added for multiple retrievers
                        subset_columns = []
                        llm_filter_response[m] = {}

                        for i, llm in enumerate(flow_config.llm_retrieval):
                            output = self.LLM_call(state, flow_config, llm)

                            if (
                                flow_config.retrieval_type == SettingKey.HYBRID_RETRIEVAL
                                and output.response != "null"
                            ):
                                subset_columns.extend(
                                    SchemaRetriever.extract_valid_columns(output.response)
                                )
                                llm_filter_response[m][i] = output

                            elif (
                                flow_config.retrieval_type == SettingKey.LLM_RETRIEVAL
                                and output.group != "null"
                            ):
                                subset_columns.extend(output.group)
                        subset_columns = list(set(subset_columns))
                    response[m] = RetrieverResponse(group=subset_columns, coi=state.example.cois[m])
                # Value ambiguity
                elif flow_config.ac_type in (
                    SettingKey.VALUE_AMBIGUITY_SIGL_COL,
                    SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
                ):
                    state.example.coi = state.example.cois[m]
                    state.example.coi_table = next(iter(state.parser_result["selected"][m]))
                    subset_columns, coi, voi = self.get_coi_values(
                        state, flow_config.ac_type, flow_config.similarity_criteria
                    )
                    subset_columns = subset_columns if len(subset_columns) > 1 else []
                    state.example.voi = voi
                    state.example.subset_columns = subset_columns
                    response[m] = (
                        RetrieverSingleColValueResponse(group=subset_columns, coi=coi, voi=voi)
                        if flow_config.ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL
                        else RetrieverMultiColValueResponse(group=subset_columns, coi=coi, voi=voi)
                    )

        return {self.output_key: response, self.clss_output_key: llm_filter_response}

    def get_coi_values(self, state, ac_type, similarity_criteria):

        voi = state.example.coi[1]
        coi, coi_value = self._get_coi_values(voi, state, ac_type, similarity_criteria)
        # db_values = self.col_val_table[state.example.db_id][similarity_criteria]
        return coi_value, coi, voi

    def _get_coi_values(self, voi, state, ac_type, similarity_criteria):
        """
        Return coi and coi_value based on voi and similarity_criteria
        """
        schema_values = state.example.schema_values
        schema_values = lower_keys(schema_values)
        table_names = [k for d in state.parser_result["selected"] for k, v in d.items()]
        table_name = state.example.coi_table or table_names[0]
        coi = state.example.coi[0].lower()
        if ac_type == SettingKey.VALUE_AMBIGUITY_SIGL_COL:
            # For single column value ambiguity,
            # return all unique values in that column
            coi_value = schema_values.get(table_name, {}).get(coi, [])
        elif ac_type == SettingKey.VALUE_AMBIGUITY_MULTI_COLS:
            if similarity_criteria == SettingKey.EXACT_MATCHES:
                matched_columns = self._get_exact_value_matches(
                    voi, coi, {table_name: schema_values.get(table_name, {})}
                )
            elif similarity_criteria == SettingKey.RANGE_MATCHES:
                matched_columns = self._get_range_value_matches(
                    voi, coi, {table_name: schema_values.get(table_name, {})}
                )
            else:
                raise ValueError(
                    f"Unsupported similarity_criteria for value ambiguity: {similarity_criteria}"
                )
            # coi = lower_all(db_values[voi])
            # For multi column value ambiguity,
            # return all columns that have this value
            coi_value = matched_columns
        else:
            raise ValueError(f"Unsupported ac_type for value ambiguity: {ac_type}")
        return coi, coi_value

    def _get_exact_value_matches(self, value, coi, tables_values):
        """
        This function gets the columns that contain the value.
        """
        col_matches = []
        for table, columns in tables_values.items():
            if coi in columns:
                for column, values in columns.items():
                    if str(value) in [str(v) for v in values]:
                        col_matches.append(column)
        return col_matches

    def _get_range_value_matches(self, value, coi, tables_values):
        """
        This function gets columns that contain values in the range of the given value.
        """
        col_matches = []
        for table, columns in tables_values.items():
            if coi in columns:
                numeric_ranges = {}
                date_ranges = {}
                for col, values in columns.items():
                    clean_values = [v for v in values if v not in (None, "", "NA")]
                    numeric_vals = [float(v) for v in clean_values if is_number(v)]
                    date_vals = [int(v) for v in clean_values if is_date(v) and str(v).isdigit()]
                    if numeric_vals:
                        numeric_ranges[col] = (min(numeric_vals), max(numeric_vals))
                    if date_vals:
                        date_ranges[col] = (min(date_vals), max(date_vals))
                if is_number(value):
                    val = float(value)
                    for col, (low, high) in numeric_ranges.items():
                        if low <= val <= high:
                            col_matches.append(col)
                elif is_date(value) and str(value).isdigit():
                    # TODO: Regarding current is_number check, this is_date will never be called.
                    # TODO: But, for fdi, this is_date would be useful, so keep it here.
                    try:
                        if isinstance(value, int):
                            value_ts = pd.Timestamp(year=value, month=1, day=1)
                        else:
                            value_ts = pd.to_datetime(str(value), errors="coerce")
                    except Exception:
                        return []

                    for col, (low, high) in date_ranges.items():
                        low = pd.to_datetime(str(low), errors="coerce")
                        high = pd.to_datetime(str(high), errors="coerce")
                        if low <= value_ts <= high:
                            col_matches.append(col)
        return col_matches


def is_number(value):
    """Check if a value is numeric."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_date(value):
    """Check if a value looks like a year or a date."""
    try:
        if isinstance(value, (int, float)):
            return 1900 <= int(value) <= 2100
        datetime.strptime(str(value), "%Y-%m-%d")
        return True
    except Exception:
        return False


def lower_keys(d):
    if isinstance(d, dict):
        return {k.lower(): lower_keys(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [lower_keys(i) for i in d]
    else:
        return d


def lower_all(d):
    if isinstance(d, dict):
        return {k.lower(): lower_all(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [lower_all(i) for i in d]
    elif isinstance(d, str):
        return d.lower()
    else:
        return d
