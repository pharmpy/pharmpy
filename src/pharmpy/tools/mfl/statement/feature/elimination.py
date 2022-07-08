from dataclasses import dataclass
from typing import List, Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature
from .symbols import Name, Wildcard


@dataclass
class Elimination(ModelFeature):
    modes: Union[List[Name[Literal['FO', 'ZO', 'MM', 'MIX-FO-MM']]], Wildcard]


class EliminationInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return Elimination(*children)

    def elimination_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def elimination_wildcard(self, tree):
        return Wildcard()
