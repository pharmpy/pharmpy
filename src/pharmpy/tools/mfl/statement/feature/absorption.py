from dataclasses import dataclass
from typing import Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

ABSORPTION_WILDCARD = tuple([Name(x) for x in ('FO', 'ZO', 'SEQ-ZO-FO', 'INST')])


@dataclass(frozen=True)
class Absorption(ModelFeature):
    modes: Union[tuple[Name[Literal['FO', 'ZO', 'SEQ-ZO-FO', 'INST']], ...], Wildcard]

    def __add__(self, other):
        if isinstance(self.modes, Wildcard) or isinstance(other.modes, Wildcard):
            return Absorption(Wildcard())
        else:
            return Absorption(
                tuple(set(self.modes + tuple([a for a in other.modes if a not in self.modes])))
            )

    def __sub__(self, other):
        if isinstance(other.modes, Wildcard):
            return Absorption((Name('INST')))
        elif isinstance(self.modes, Wildcard):
            default = ABSORPTION_WILDCARD
            all_modes = tuple([a for a in default if a not in other.modes])
        else:
            # NOTE : WILDCARD should not be used here to future proof the method
            all_modes = tuple(set([a for a in self.modes if a not in other.modes]))

        if len(all_modes) == 0:
            all_modes = (Name('INST'),)

        return Absorption(all_modes)

    def __eq__(self, other):
        if isinstance(other, Absorption):
            return set(self.modes) == set(other.modes)
        else:
            return False

    def __len__(self):
        return len(self.eval.modes)

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            return Absorption(ABSORPTION_WILDCARD)
        else:
            return self


class AbsorptionInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(Absorption, children)

    def absorption_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def absorption_wildcard(self, tree):
        return Wildcard()
