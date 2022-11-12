from dataclasses import dataclass
from typing import Tuple

from .count_interpreter import CountInterpreter
from .feature import ModelFeature, feature


@dataclass(frozen=True)
class Peripherals(ModelFeature):
    counts: Tuple[int, ...]


class PeripheralsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(Peripherals, children)
