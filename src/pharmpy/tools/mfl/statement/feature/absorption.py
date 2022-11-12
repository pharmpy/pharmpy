from dataclasses import dataclass
from typing import Literal, Tuple, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard


@dataclass(frozen=True)
class Absorption(ModelFeature):
    modes: Union[Tuple[Name[Literal['FO', 'ZO', 'SEQ-FO-ZO']], ...], Wildcard]


class AbsorptionInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(Absorption, children)

    def absorption_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def absorption_wildcard(self, tree):
        return Wildcard()
