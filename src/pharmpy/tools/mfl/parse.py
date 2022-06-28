from typing import List

from lark import Lark

from .grammar import grammar
from .interpreter import MFLInterpreter
from .statement.statement import Statement


def parse(code: str) -> List[Statement]:
    parser = Lark(
        grammar,
        # parser='lalr',
        # lexer='standard',
        # propagate_positions=False,
        # maybe_placeholders=False,
        # cache=True,
    )

    tree = parser.parse(code)

    return MFLInterpreter().interpret(tree)
