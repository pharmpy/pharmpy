from dataclasses import dataclass
from typing import Literal, Tuple, Union

from .count_interpreter import CountInterpreter
from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

PERIPHERALS_MODES_WILDCARD = tuple([Name(x) for x in ('DRUG', 'MET')])


@dataclass(frozen=True)
class Peripherals(ModelFeature):
    counts: Tuple[int, ...]
    modes: Union[Tuple[Name[Literal['DRUG', 'MET']], ...], Wildcard] = (Name('DRUG'),)

    def __add__(self, other):
        return Peripherals(
            tuple(set(self.counts + other.counts)), tuple(set(self.modes + other.modes))
        )

    def __sub__(self, other):
        all_counts = tuple([a for a in self.counts if a not in other.counts])
        all_modes = tuple([a for a in self.modes if a not in other.modes])

        if len(all_counts) == 0:
            all_counts = (0,)
        if len(all_modes) == 0:
            all_modes = (Name('DRUG'),)

        return Peripherals(all_counts, all_modes)

    def __eq__(self, other):
        if isinstance(other, Peripherals):
            return set(self.counts) == set(other.counts) and set(self.modes) == set(other.modes)
        else:
            return False

    def __len__(self):
        self_eval = self.eval
        return len(self_eval.modes) * len(self_eval.counts)

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            return Peripherals(self.counts, PERIPHERALS_MODES_WILDCARD)
        else:
            return self


class PeripheralsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        return feature(Peripherals, children)

    def peripheral_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def peripheral_wildcard(self, tree):
        return Wildcard()
