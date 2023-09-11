from dataclasses import dataclass
from typing import Literal, Tuple, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard


@dataclass(frozen=True)
class LagTime(ModelFeature):
    modes: Union[Tuple[Name[Literal['ON', 'OFF']], ...], Wildcard]
    pass


class LagTimeInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(LagTime, children)

    def lagtime_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def lagtime_wildcard(self, tree):
        return Wildcard()
