from dataclasses import dataclass
from typing import List, Literal, Union

from .count_interpreter import CountInterpreter
from .feature import ModelFeature
from .symbols import Name, Wildcard


@dataclass
class Transits(ModelFeature):
    counts: List[int]
    depot: Union[Name[Literal['DEPOT']], Name[Literal['NODEPOT']], Wildcard] = Name('DEPOT')


class TransitsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        return Transits(*children)

    def depot_option(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value.upper()
        assert value in ['*', 'DEPOT', 'NODEPOT']
        return Wildcard() if value == '*' else Name(value)
