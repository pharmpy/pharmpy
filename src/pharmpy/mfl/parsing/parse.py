from lark import Lark

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

    tree = parser.parse(code)

    definitions = find_definitions(tree)
    features = MFLInterpreter(definitions).interpret(tree)
    features = _flatten(features)

    return features


def find_definitions(tree):
    definition_trees = tree.find_data('definition')

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
