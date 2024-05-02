from dataclasses import dataclass
from typing import Literal, Tuple, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

DIRECT_EFFECT_WILDCARD = tuple([Name(x) for x in ('LINEAR', 'EMAX', 'SIGMOID')])


@dataclass(frozen=True)
class DirectEffect(ModelFeature):
    modes: Union[Tuple[Name[Literal['LINEAR', 'EMAX', 'SIGMOID']], ...], Wildcard]

    def __add__(self, other):
        if isinstance(self.modes, Wildcard) or isinstance(other.modes, Wildcard):
            return DirectEffect(Wildcard())
        else:
            return DirectEffect(
                tuple(set(self.modes + tuple([a for a in other.modes if a not in self.modes])))
            )

    def __sub__(self, other):
        if isinstance(other, DirectEffect):
            if isinstance(other.modes, Wildcard):
                return None
            elif isinstance(self.modes, Wildcard):
                default = DIRECT_EFFECT_WILDCARD
                all_modes = tuple([a for a in default if a not in other.modes])
            else:
                # NOTE : WILDCARD should not be used here to future proof the method
                all_modes = tuple(set([a for a in self.modes if a not in other.modes]))

            if len(all_modes) == 0:
                all_modes = None

            return DirectEffect(all_modes)
        else:
            return self

    def __eq__(self, other):
        if isinstance(other, DirectEffect):
            return set(self.modes) == set(other.modes)
        else:
            return False

    def __len__(self):
        return len(self.eval.modes)

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            return DirectEffect(DIRECT_EFFECT_WILDCARD)
        else:
            return self


class DirectEffectInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(DirectEffect, children)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def pdtype_wildcard(self, tree):
        return Wildcard()
