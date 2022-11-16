from dataclasses import dataclass
from typing import Literal, Tuple, Union

from .count_interpreter import CountInterpreter
from .feature import ModelFeature, feature
from .symbols import Name, Wildcard


@dataclass(frozen=True)
class Transits(ModelFeature):
    counts: Tuple[int, ...]
    depot: Union[Name[Literal['DEPOT']], Name[Literal['NODEPOT']], Wildcard] = Name('DEPOT')


class TransitsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        return feature(Transits, children)

    def depot_option(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value.upper()
        assert value in ['*', 'DEPOT', 'NODEPOT']
        return Wildcard() if value == '*' else Name(value)
