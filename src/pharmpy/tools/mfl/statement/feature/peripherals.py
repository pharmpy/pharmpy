from dataclasses import dataclass
from typing import List

from .count_interpreter import CountInterpreter
from .feature import ModelFeature


@dataclass
class Peripherals(ModelFeature):
    counts: List[int]


class PeripheralsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return Peripherals(children[0])
