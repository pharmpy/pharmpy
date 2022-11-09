from typing import List

from lark import Lark

from .grammar import grammar
from .interpreter import MFLInterpreter
from .statement.statement import Statement


def parse(code: str) -> List[Statement]:
    parser = Lark(
        grammar,
        start='start',
        parser='lalr',
        # lexer='standard',  # NOTE This does not work because lexing for the
        #      MFL grammar is context-dependent
        propagate_positions=False,
        maybe_placeholders=False,
        debug=False,
        cache=True,
    )

    tree = parser.parse(code)

    return MFLInterpreter().interpret(tree)
