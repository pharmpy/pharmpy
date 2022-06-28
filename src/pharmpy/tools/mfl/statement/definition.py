from dataclasses import dataclass
from typing import List

from lark.visitors import Interpreter


@dataclass
class Definition:
    name: str
    value: List[str]


class DefinitionInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 2
        return Definition(*children)

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()
