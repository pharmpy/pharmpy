from dataclasses import dataclass
from typing import Literal, Union

from .count_interpreter import CountInterpreter
from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

TRANSITS_DEPOT_WILDCARD = tuple([Name(x) for x in ('DEPOT', 'NODEPOT')])


@dataclass(frozen=True)
class Transits(ModelFeature):
    counts: tuple[int, ...]
    depot: Union[tuple[Name[Literal['DEPOT', 'NODEPOT']], ...], Wildcard] = (Name('DEPOT'),)

    def __add__(self, other):
        # NOTE : Use with caution as no addition logic is implemented here
        if isinstance(self.depot, Wildcard) or isinstance(other.depot, Wildcard):
            depot = Wildcard()
        else:
            depot = tuple(set(self.depot + tuple([a for a in other.depot if a not in self.depot])))

        counts = tuple(set(self.counts + tuple([a for a in other.counts if a not in self.counts])))

        return Transits(counts=counts, depot=depot)

    def __sub__(self, other):
        # NOTE : Use with caution as no subtraction logic is implemented here
        if isinstance(other.depot, Wildcard):
            all_depot = (Name('DEPOT'),)
        elif isinstance(self.depot, Wildcard):
            default = TRANSITS_DEPOT_WILDCARD
            all_depot = tuple([a for a in default if a not in other.depot])
        else:
            all_depot = tuple([a for a in self.depot if a not in other.depot])

        if len(all_depot) == 0:
            all_depot = (Name('DEPOT'),)

        all_counts = tuple([a for a in self.counts if a not in other.counts])

        if len(all_counts) == 0:
            all_counts = (0,)

        return Transits(counts=all_counts, depot=all_depot)

    def __eq__(self, other):
        if isinstance(other, Transits):
            if isinstance(self.depot, Wildcard):
                lhs_depot = self.depot
            else:
                lhs_depot = set(self.depot)

            if isinstance(other.depot, Wildcard):
                rhs_depot = other.depot
            else:
                rhs_depot = set(other.depot)

            return (set(self.counts) == set(other.counts), lhs_depot == rhs_depot)
        else:
            return False

    def __len__(self):
        self_eval = self.eval
        transit_len = len(self_eval.counts) * len(self_eval.depot)
        return transit_len

    @property
    def eval(self):
        if isinstance(self.depot, Wildcard):
            return Transits(self.counts, TRANSITS_DEPOT_WILDCARD)
        else:
            return self


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
