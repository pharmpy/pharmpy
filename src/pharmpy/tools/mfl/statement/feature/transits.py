from dataclasses import dataclass
from typing import Literal, Tuple, Union

from .count_interpreter import CountInterpreter
from .feature import ModelFeature, feature
from .symbols import Name, Wildcard


@dataclass(frozen=True)
class Transits(ModelFeature):
    counts: Tuple[int, ...]
    depot: Union[Tuple[Name[Literal['DEPOT', 'NODEPOT']], ...], Wildcard] = (Name('DEPOT'),)


class TransitsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        return feature(Transits, children)

    def depot_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def depot_wildcard(self, tree):
        return Wildcard()
