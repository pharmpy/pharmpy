from dataclasses import dataclass
from typing import Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

LAGTIME_WILDCARD = tuple([Name(x) for x in ('ON', 'OFF')])


@dataclass(frozen=True)
class LagTime(ModelFeature):
    modes: Union[tuple[Name[Literal['ON', 'OFF']], ...], Wildcard]

    def __add__(self, other):
        if isinstance(self.modes, Wildcard) or isinstance(other.modes, Wildcard):
            return LagTime(Wildcard())
        else:
            return LagTime(
                tuple(set(self.modes + tuple([a for a in other.modes if a not in self.modes])))
            )

    def __sub__(self, other):
        if isinstance(other.modes, Wildcard):
            return LagTime((Name('OFF')))
        elif isinstance(self.modes, Wildcard):
            default = LAGTIME_WILDCARD
            all_modes = tuple([a for a in default if a not in other.modes])
        else:
            all_modes = tuple([a for a in self.modes if a not in other.modes])

        if len(all_modes) == 0:
            all_modes = (Name('OFF'),)

        return LagTime(all_modes)

    def __eq__(self, other):
        if isinstance(other, LagTime):
            return set(self.modes) == set(other.modes)
        else:
            return False

    def __len__(self):
        return len(self.eval.modes)

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            return LagTime(LAGTIME_WILDCARD)
        else:
            return self


class LagTimeInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(LagTime, children)

    def lagtime_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def lagtime_wildcard(self, tree):
        return Wildcard()
