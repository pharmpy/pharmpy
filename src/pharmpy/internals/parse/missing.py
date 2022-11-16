from typing import Mapping, Tuple, Union

from lark import Tree, Visitor
from lark.lexer import Token


def _rule_of(item: Union[Tree, Token]) -> str:
    if isinstance(item, Tree):
        return item.data
    else:
        return item.type


def _empty_node(rule: str):
    """Create empty Tree or Token, depending on (all) upper-case or not."""
    return Token(rule, '') if rule == rule.upper() else Tree(rule, [])


class InsertMissing(Visitor):
    def __init__(self, missing: Tuple[Mapping[str, Tuple[Tuple[int, str], ...]], ...]):
        self._missing = missing

    def __default__(self, tree: Tree):
        for d in self._missing:
            try:
                children = d[tree.data]
            except KeyError:
                pass
            else:
                for pos, name in children:
                    if not any(_rule_of(child) == name for child in tree.children):
                        tree.children.insert(pos, _empty_node(name))
