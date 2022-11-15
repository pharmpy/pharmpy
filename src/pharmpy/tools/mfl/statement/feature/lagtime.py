from dataclasses import dataclass

from lark.visitors import Interpreter

from .feature import ModelFeature


@dataclass(frozen=True)
class LagTime(ModelFeature):
    pass


class LagTimeInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert not children
        return LagTime()
