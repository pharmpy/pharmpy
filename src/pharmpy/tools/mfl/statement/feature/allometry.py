from dataclasses import dataclass
from typing import Optional, Union

from lark.visitors import Interpreter

from pharmpy.model import Model

from .feature import ModelFeature
from .symbols import Symbol


@dataclass(frozen=True)
class Allometry(ModelFeature):
    covariate: Union[Symbol, str]
    reference: float = 70.0

    def eval(self, model: Optional[Model] = None):
        return None


class AllometryInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        return Allometry(covariate=children[0], reference=children[1])

    def value(self, tree):
        return tree.children[0].value

    def decimal(self, tree):
        return float(tree.children[0].value)
