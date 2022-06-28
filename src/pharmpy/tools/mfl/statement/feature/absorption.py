from dataclasses import dataclass
from typing import List, Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature
from .symbols import Name, Wildcard


@dataclass
class Absorption(ModelFeature):
    modes: Union[List[Name[Literal['FO', 'ZO', 'SEQ-FO-ZO']]], Wildcard]


class AbsorptionInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return Absorption(*children)

    def absorption_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def absorption_wildcard(self, tree):
        return Wildcard()
