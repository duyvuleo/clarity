import logging
import random
from typing import Any
import re


from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from clarity.generation.task_flows.components.data_types import LOGGER_NAME, SettingKey
from clarity.generation.task_flows.components.nodes.base import UtteranceNodeBase
from clarity.generation.task_flows.components.states import ConversationState
from clarity.generation.task_flows.components.signatures import ParserResponse
import sqlglot
from sqlglot import exp


from clarity.utils.terminal import Color

logger = logging.getLogger(LOGGER_NAME)


class SQLParser(UtteranceNodeBase):
    def __init__(self, output_pydantic_object: BaseModel):
        self.output_pydantic_object = output_pydantic_object
        self.output_key: str = "parser_result"

    # extract column value pairs
    @staticmethod
    def extract_column_value_pairs(sql):
        # Step 1: Build alias -> table name map
        alias_to_table = {}
        table_names = set()

        for table in sql.find_all(exp.Table):
            alias = table.alias_or_name
            real_name = table.name
            alias_to_table[alias] = real_name
            table_names.add(real_name)

        is_single_table = len(table_names) == 1
        single_table_name = next(iter(table_names)) if is_single_table else None

        # Step 2: Extract column-value pairs grouped by table name
        table_column_values = {}

        # Supported binary comparison operators
        supported_ops = {
            exp.EQ: "=",
            exp.NEQ: "!=",
            exp.GT: ">",
            exp.LT: "<",
            exp.GTE: ">=",
            exp.LTE: "<=",
        }

        def process_condition(expr):
            if isinstance(expr, exp.And):
                process_condition(expr.left)
                process_condition(expr.right)
            else:
                for op_class, op_symbol in supported_ops.items():
                    if isinstance(expr, op_class):
                        left = expr.args.get("this")
                        right = expr.args.get("expression")

                        def add_pair(column_expr, literal_expr):
                            if isinstance(column_expr, exp.Column) and isinstance(
                                literal_expr, exp.Literal
                            ):
                                col_name = column_expr.name
                                col_table_alias = column_expr.table

                                # Determine actual table name
                                if col_table_alias:
                                    table_name = alias_to_table.get(col_table_alias)
                                elif is_single_table:
                                    table_name = single_table_name
                                else:
                                    table_name = None  # Ambiguous

                                if table_name:
                                    table_column_values.setdefault(table_name, []).append(
                                        (col_name, literal_expr.name)
                                    )

                        add_pair(left, right)
                        add_pair(right, left)

        # Process WHERE clause
        where_expr = sql.args.get("where")
        if where_expr:
            process_condition(where_expr.this)

        # Process JOIN ON conditions
        for join in sql.find_all(exp.Join):
            on_expr = join.args.get("on")
            if on_expr:
                process_condition(on_expr)

        return table_column_values

    # extract columns with None values
    @staticmethod
    def extract_columns(sql):
        # Step 1: Collect all table references (including unaliased ones)
        alias_to_table = {}
        table_names = set()

        for table in sql.find_all(exp.Table):
            alias = table.alias
            name = table.name
            table_names.add(name)
            if alias:
                alias_to_table[alias] = name
            else:
                alias_to_table[name] = name  # no alias, use table name directly

        is_single_table = len(table_names) == 1
        single_table_name = next(iter(table_names)) if is_single_table else None

        # Step 2: Collect columns by real table name
        table_columns = {}

        for column in sql.find_all(exp.Column):
            col_name = column.name
            table_alias = column.table  # might be None

            if table_alias:  # qualified
                real_table = alias_to_table.get(table_alias)
                if real_table:
                    table_columns.setdefault(real_table, set()).add(col_name)
            elif is_single_table:  # unqualified and only one table
                table_columns.setdefault(single_table_name, set()).add(col_name)
            # else: unqualified and multi-table – ambiguous, so skip

        table_columns = {table: sorted(cols) for table, cols in table_columns.items()}

        # Step 3: Extract columns involved in 'join' clause
        join_columns = {}

        for join in sql.find_all(exp.Join):
            on_condition = join.args.get("on")
            if on_condition:
                for comparison in on_condition.find_all(exp.EQ):
                    left = comparison.args.get("this")
                    right = comparison.args.get("expression")

                    if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                        # Resolve real table names (not aliases)
                        left_table = alias_to_table.get(left.table)
                        right_table = alias_to_table.get(right.table)

                        if left_table and right_table:
                            # Add column names to the appropriate table
                            join_columns.setdefault(left_table, set()).add(left.name)
                            join_columns.setdefault(right_table, set()).add(right.name)

        join_columns = {table: sorted(cols) for table, cols in join_columns.items()}
        return SQLParser.remove_join_columns(table_columns, join_columns)

    @staticmethod
    def select_random_table_column(table_column_dict, m):
        flattened = [(key, val) for key, vals in table_column_dict.items() for val in vals]
        sampled = random.sample(flattened, m)

        selected = []
        for idx, (k, v) in enumerate(sampled):
            selected.append({k: v})

        """
        if not table_column_dict:
            return None  # or raise an error

        # Step 1: Randomly select a key (table)
        table = random.choice(list(table_column_dict.keys()))
        table = list(table_column_dict.keys())
        
        # Step 2: Randomly select a column from that table's column list
        columns = table_column_dict[table]
        if not columns:
            return None  # or raise an error

        column = random.choice(columns)
        return {table: column}
        """
        return selected

    # @staticmethod
    # def lowercase_keys(d):
    #    return {str(k).lower(): v for k, v in d.items()}

    @staticmethod
    def remove_join_columns(d1, d2):
        result = {}

        for key, values in d1.items():
            # Remove values only if the key exists in d2
            to_remove = set(d2.get(key, []))
            new_values = [v for v in values if v not in to_remove]
            # Keep key only if more than one value remains
            if len(new_values) > 0:
                result[key] = new_values

        return result

    @staticmethod
    def remove_value_by_key(t1, t2):
        new_t2 = {}
        for key, value in t2.items():
            if key in t1:
                t1_val = t1[key]
                # If value is a list, remove the matching item
                if isinstance(value, list):
                    new_list = [v for v in value if v != t1_val]
                    # if the list is not empty
                    if len(new_list) > 0:
                        new_t2[key] = new_list
            else:
                new_t2[key] = value
        return new_t2

    @staticmethod
    def replace_double_quotes_in_where_literals(query: str) -> str:

        query = query.replace("`", '"')
        # Regex pattern to isolate the WHERE clause
        where_pattern = re.compile(
            r"(WHERE\s+.*?)(GROUP BY|ORDER BY|LIMIT|$)", re.IGNORECASE | re.DOTALL
        )

        # List of comparison operators to support
        operators = ["=", "!=", "<=", ">=", "<", ">"]

        # Create a combined regex pattern to find: operator + space + "quoted string"
        operator_pattern = "|".join(re.escape(op) for op in operators)
        literal_pattern = rf'({operator_pattern})\s*"([^"]*)"'

        def process_where_clause(match):
            where_clause = match.group(1)
            suffix = match.group(2)

            # Replace all matched operators with single-quoted literals
            updated_where = re.sub(literal_pattern, r"\1 '\2'", where_clause)

            return updated_where + suffix

        # Apply the transformation to the full query
        updated_query = where_pattern.sub(process_where_clause, query)

        return updated_query

    @staticmethod
    def lowercase_keys(d):
        if isinstance(d, dict):
            return {k.lower(): SQLParser.lowercase_keys(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [SQLParser.lowercase_keys(item) for item in d]
        else:
            return d

    @staticmethod
    def check_col_vals(state, selected, m):
        column_vals = []
        for i in range(m):
            key = next(iter(selected[i]))
            table_name = next(iter(state.example.cois_pair[i]))
            column_name = next(iter(state.example.cois_pair[i][key])).lower()
            if (
                state.example.schema_values.get(table_name)
                and column_name in state.example.schema_values[table_name]
            ):
                column_vals.append(state.example.schema_values[table_name][column_name])
            else:
                column_vals.append([])
        return column_vals

    def __call__(self, state: ConversationState, config: RunnableConfig) -> Any:
        logger.debug(Color.YELLOW(f"[Node]: ## {self.__class__.__name__} ##"))
        flow_config = self._get_flow_config(config)
        # specific to FDI manual only - select the first query
        if isinstance(state.example.expected_query, list):
            state.example.expected_query = state.example.expected_query[0]

        parsed_query = sqlglot.parse_one(
            SQLParser.replace_double_quotes_in_where_literals(state.example.expected_query)
        )
        value_pairs = SQLParser.extract_column_value_pairs(parsed_query)
        columns = SQLParser.extract_columns(parsed_query)
        if state.example.db_id is not None:
            columns, value_pairs = (
                SQLParser.lowercase_keys(columns),
                SQLParser.lowercase_keys(value_pairs),
            )

        # discarding columns that were selected in previous round but were not successful in a/c term generation
        if state.parser_result:
            ind = []
            if state.example.retriever_decision is not None:
                ind = [
                    i
                    for i in range(len(state.example.retriever_decision))
                    if state.example.retriever_decision[i] == False
                ]

            if state.example.term_decision is not None:
                ind = ind + [
                    i
                    for i in range(len(state.example.term_decision))
                    if state.example.term_decision[i] == False
                ]

            if state.example.to_remove is None:
                state.example.to_remove = [state.parser_result["selected"][i] for i in ind]
            else:
                state.example.to_remove = state.example.to_remove + [
                    state.parser_result["selected"][i] for i in ind
                ]

            for i in range(len(state.example.to_remove)):
                if (
                    flow_config.ac_type
                    not in [
                        SettingKey.VALUE_AMBIGUITY_MULTI_COLS,
                        SettingKey.VALUE_AMBIGUITY_SIGL_COL,
                    ]
                    and flow_config.ac_type != SettingKey.VALUE_CONFUSION
                ):
                    columns = SQLParser.remove_value_by_key(state.example.to_remove[i], columns)
                else:
                    value_pairs = SQLParser.remove_value_by_key(
                        state.example.to_remove[i], value_pairs
                    )

        if (
            SettingKey.COLUMN_AMBIGUITY == flow_config.ac_type
            or SettingKey.COLUMN_CONFUSION == flow_config.ac_type
        ):
            m = min(flow_config.multifacet, sum([len(columns[db]) for db in columns]))
            selected = SQLParser.select_random_table_column(columns, m)
            state.example.cois = [next(iter(d.values())) for d in selected]
        else:
            state.example.schema_values = SQLParser.lowercase_keys(state.example.schema_values)
            m = min(flow_config.multifacet, sum([len(value_pairs[db]) for db in value_pairs]))
            selected = SQLParser.select_random_table_column(value_pairs, m)
            state.example.cois_pair = selected
            state.example.cois = [next(iter(d.values())) for d in selected]
            # this is to check whether whether column values exist in the database or not
            state.example.column_values = SQLParser.check_col_vals(state, selected, m)
            m = sum(1 for sublist in state.example.column_values if sublist)

        # this is for the case where multifacet columns e.g., 2 columns do not exist
        response = (
            ParserResponse(columns=columns, value_pairs=value_pairs, selected=selected)
            if m == flow_config.multifacet
            else ParserResponse()
        )

        return {self.output_key: response}
