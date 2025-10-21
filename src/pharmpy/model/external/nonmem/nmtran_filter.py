from dataclasses import dataclass
from typing import Iterable, Literal, Type, Union, cast

from lark import Lark, Token

from pharmpy.deps import pandas as pd

FilterOperator = Union[
    Literal["OP_EQ"],
    Literal["OP_NE"],
    Literal["OP_LT"],
    Literal["OP_GT"],
    Literal["OP_LT_EQ"],
    Literal["OP_GT_EQ"],
    Literal["OP_STR_EQ"],
    Literal["OP_STR_NE"],
]


def operator_type(op: FilterOperator):
    return str if op.startswith("OP_STR_") else float


def operator_symbol(op: FilterOperator) -> str:
    if op == "OP_EQ":
        return "=="
    elif op == "OP_NE":
        return "!="
    elif op == "OP_LT":
        return "<"
    elif op == "OP_GT":
        return ">"
    elif op == "OP_LT_EQ":
        return "<="
    elif op == "OP_GT_EQ":
        return ">="
    elif op == "OP_STR_EQ":
        return "=="
    elif op == "OP_STR_NE":
        return "!="


@dataclass(frozen=True)
class Filter:
    column: str
    operator: FilterOperator
    expr: str


def filter_column(f: Filter) -> str:
    return f.column


def _parse_filter_statement(parser: Lark, s: str):
    tree = parser.parse(s)
    column = None
    operator = "OP_STR_EQ"
    expr = None
    for st in tree.iter_subtrees():
        if st.data == "column":
            column = str(st.children[0])
        elif st.data == "expr":
            expr = str(st.children[0])
        elif st.data == "operator":
            operator_token = cast(Token, st.children[0])
            operator = operator_token.type

    assert column is not None
    assert operator is not None
    assert expr is not None

    if len(expr) >= 3 and (
        (expr.startswith("'") and expr.endswith("'"))
        or (expr.startswith('"') and expr.endswith('"'))
    ):
        expr = expr[1:-1]

    return Filter(column=column, operator=cast(FilterOperator, operator), expr=expr)


def _parser():
    _grammar = r"""
        start: column skip1? (operator skip2?)? expr
        column: COLNAME
        COLNAME: /\w+/
        skip1: WS
        skip2: WS
        WS: /\s+/
        operator: OP_EQ | OP_STR_EQ | OP_NE | OP_STR_NE | OP_LT | OP_GT | OP_LT_EQ | OP_GT_EQ
        OP_EQ    : ".EQN."
        OP_STR_EQ: ".EQ." | "==" | "="
        OP_NE    : ".NEN."
        OP_STR_NE: ".NE." | "/="
        OP_LT    : ".LT." | "<"
        OP_GT    : ".GT." | ">"
        OP_LT_EQ : ".LE." | "<="
        OP_GT_EQ : ".GE." | ">="
        expr: EXPR | QEXPR
        EXPR  : /[^"',;()=<>\/.\s][^"',;()=\s]*/
        QEXPR : /"[^"]*"/
                | /'[^']*'/
    """
    return Lark(
        _grammar,
        start="start",
        parser="lalr",
        lexer="contextual",
        propagate_positions=False,
        maybe_placeholders=False,
        debug=False,
        cache=True,
    )


def parse_filter_statements(statements: Iterable[str]):
    p = _parser()
    for s in statements:
        yield _parse_filter_statement(p, s)


def _filter_to_expr(filter: Filter):
    _operator_type = operator_type(filter.operator)
    _operator = operator_symbol(filter.operator)

    if _operator_type is str:
        return f'({filter.column} {_operator} "{filter.expr}")'
    else:
        return f"({filter.column} {_operator} {filter.expr})"


def _negation(expression: str):
    return "not" + expression


def _conjunction(expressions: Iterable[str]):
    return " & ".join(expressions)


def filter_update_in_place(df: pd.DataFrame, res: pd.Series):
    view = df.loc[res]
    df._update_inplace(view)


def filter_dataset(
    df: pd.DataFrame,
    filters: Iterable[Filter],
    negate: bool,
):
    expressions = map(_filter_to_expr, filters)

    if negate:
        expressions = map(_negation, expressions)

    expr = _conjunction(expressions)

    return cast(
        pd.Series,
        df.eval(
            expr,
            parser="pandas",  # NOTE: Default, but it is better to be explicit.
            engine="numexpr",  # NOTE: This is not guaranteed, but it is better to be explicit.
            # SEE: https://github.com/pandas-dev/pandas/blob/9c8bc3e55188c8aff37207a74f1dd144980b8874/pandas/core/computation/eval.py#L341-L358  # noqa: E501
            local_dict={},  # NOTE: We do not allow access to locals.
            # SEE: https://github.com/pandas-dev/pandas/blob/6e27e26b2f25ff98cea60d536b6d4a53e2f0a17d/pandas/core/computation/scope.py#L172  # noqa: E501
            global_dict={},  # NOTE: We do not allow access to globals.
            # SEE: https://github.com/pandas-dev/pandas/blob/6e27e26b2f25ff98cea60d536b6d4a53e2f0a17d/pandas/core/computation/scope.py#L177  # noqa: E501
            # NOTE: Some default "globals" are included anyway.
            # SEE: https://github.com/pandas-dev/pandas/blob/6e27e26b2f25ff98cea60d536b6d4a53e2f0a17d/pandas/core/computation/scope.py#L91-L100  # noqa: E501
            target=None,  # NOTE: We do not allow assigning to df.
            # SEE: https://github.com/pandas-dev/pandas/blob/9c8bc3e55188c8aff37207a74f1dd144980b8874/pandas/core/computation/eval.py#L376  # noqa: E501
            inplace=False,  # NOTE: Default, but it is better to be explicit.
            # SEE: https://github.com/pandas-dev/pandas/blob/9c8bc3e55188c8aff37207a74f1dd144980b8874/pandas/core/computation/eval.py#L420-L421  # noqa: E501
            # SEE: https://github.com/pandas-dev/pandas/blob/9c8bc3e55188c8aff37207a74f1dd144980b8874/pandas/core/computation/eval.py#L377  # noqa: E501
        ),
    )


@dataclass(frozen=True)
class Block:
    operator_type: Type[str] | Type[float]
    filters: list[Filter]
    convert: list[str]


def filter_schedule(filters: Iterable[Filter]):
    it = iter(filters)
    try:
        filter = next(it)
    except StopIteration:
        return

    _parsed_for_filters = set()

    def _flush():
        yield Block(operator_type=_operator_type, filters=_filters, convert=_convert)

    while True:
        # NOTE: Initialize block.
        _operator_type = operator_type(filter.operator)
        _filters = [filter]
        _column = filter_column(filter)
        _convert = []
        if _operator_type is not str and _column not in _parsed_for_filters:
            # NOTE: Only the first filter in a numeric block can introduce a new parsed columns.
            _convert.append(_column)
            _parsed_for_filters.add(_column)

        # NOTE: Extend block.
        while True:
            try:
                filter = next(it)
            except StopIteration:
                yield from _flush()
                return

            if (
                operator_type(filter.operator) is not _operator_type
                or filter_column(filter) not in _parsed_for_filters
            ):
                yield from _flush()
                break

            _filters.append(filter)
