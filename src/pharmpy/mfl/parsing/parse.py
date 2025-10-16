from lark import Lark, UnexpectedToken

from .grammar import grammar
from .interpreters import DefinitionInterpreter, MFLInterpreter


def parse(code: str):
    parser = Lark(
        grammar,
        start='start',
        parser='lalr',
        # lexer='standard',  # NOTE: This does not work because lexing for the
        #      MFL grammar is context-dependent
        propagate_positions=False,
        maybe_placeholders=False,
        debug=False,
        cache=True,
    )

    try:
        tree = parser.parse(code)
    except UnexpectedToken as e:
        raise ValueError(f'Error in parsing: token {e.token} at line {e.line} col {e.column}')

    definitions = find_definitions(tree)
    features = MFLInterpreter(definitions).interpret(tree)
    features = _flatten(features)

    return features


def find_definitions(tree):
    definition_trees = list(tree.find_data('definition'))

    if not definition_trees:
        return dict()

    definitions = dict()
    for subtree in definition_trees:
        symbol, values = DefinitionInterpreter().interpret(subtree)
        definitions[symbol] = values

    return definitions


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result
