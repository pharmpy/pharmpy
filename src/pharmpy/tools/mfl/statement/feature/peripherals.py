from dataclasses import dataclass
from typing import Tuple

from .count_interpreter import CountInterpreter
from .feature import ModelFeature, feature


@dataclass(frozen=True)
class Peripherals(ModelFeature):
    counts: Tuple[int, ...]

    def __add__(self, other):
        return Peripherals(
            tuple(set(self.counts + tuple([a for a in other.counts if a not in self.counts])))
        )

    def __sub__(self, other):
        all_counts = tuple([a for a in self.counts if a not in other.counts])

        if len(all_counts) == 0:
            all_counts = (0,)

        return Peripherals(all_counts)

    def __eq__(self, other):
        return set(self.counts) == set(other.counts)


class PeripheralsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(Peripherals, children)
