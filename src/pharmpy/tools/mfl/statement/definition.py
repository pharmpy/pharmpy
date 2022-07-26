from dataclasses import dataclass
from typing import Tuple

from lark.visitors import Interpreter


@dataclass(frozen=True)
class Definition:
    name: str
    value: Tuple[str]


class DefinitionInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 2
        return Definition(children[0], tuple(children[1]))

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()
