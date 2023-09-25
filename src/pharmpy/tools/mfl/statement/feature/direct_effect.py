from dataclasses import dataclass
from typing import Literal, Tuple, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard


@dataclass(frozen=True)
class DirectEffect(ModelFeature):
    modes: Union[Tuple[Name[Literal['LINEAR', 'EMAX', 'SIGMOID']], ...], Wildcard]


class DirectEffectInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(DirectEffect, children)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def pdtype_wildcard(self, tree):
        return Wildcard()
