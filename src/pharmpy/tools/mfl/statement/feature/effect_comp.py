from dataclasses import dataclass
from typing import Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

EFFECTCOMP_WILDCARD = tuple([Name(x) for x in ('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN')])


@dataclass(frozen=True)
class EffectComp(ModelFeature):
    modes: Union[tuple[Name[Literal['LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN']], ...], Wildcard]

    def __add__(self, other):
        if isinstance(self.modes, Wildcard) or isinstance(other.modes, Wildcard):
            return EffectComp(Wildcard())
        else:
            return EffectComp(
                tuple(set(self.modes + tuple([a for a in other.modes if a not in self.modes])))
            )

    def __sub__(self, other):
        if isinstance(other, EffectComp):
            if isinstance(other.modes, Wildcard):
                return None
            elif isinstance(self.modes, Wildcard):
                default = EFFECTCOMP_WILDCARD
                all_modes = tuple([a for a in default if a not in other.modes])
            else:
                # NOTE : WILDCARD should not be used here to future proof the method
                all_modes = tuple(set([a for a in self.modes if a not in other.modes]))

            if len(all_modes) == 0:
                all_modes = None

            return EffectComp(all_modes)
        else:
            return self

    def __eq__(self, other):
        if isinstance(other, EffectComp):
            return set(self.eval.modes) == set(other.eval.modes)
        else:
            return False

    def __len__(self):
        return len(self.eval.modes)

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            return EffectComp(EFFECTCOMP_WILDCARD)
        else:
            return self


class EffectCompInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(EffectComp, children)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def pdtype_wildcard(self, tree):
        return Wildcard()
