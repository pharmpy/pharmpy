from dataclasses import dataclass
from typing import Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

INDIRECT_EFFECT_MODES_WILDCARD = tuple([Name(x) for x in ('LINEAR', 'EMAX', 'SIGMOID')])
INDIRECT_EFFECT_PRODUCTION_WILDCARD = tuple([Name(x) for x in ('DEGRADATION', 'PRODUCTION')])


@dataclass(frozen=True)
class IndirectEffect(ModelFeature):
    modes: Union[tuple[Name[Literal['LINEAR', 'EMAX', 'SIGMOID']], ...], Wildcard]
    production: Union[tuple[Name[Literal['DEGRADATION', 'PRODUCTION']], ...], Wildcard]

    def __len__(self):
        self_eval = self.eval
        return len(self_eval.modes) * len(self_eval.production)

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            modes = INDIRECT_EFFECT_MODES_WILDCARD
        else:
            modes = self.modes

        if isinstance(self.production, Wildcard):
            production = INDIRECT_EFFECT_PRODUCTION_WILDCARD
        else:
            production = self.production

        return IndirectEffect(modes, production)


class IndirectEffectInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 2
        return feature(IndirectEffect, children)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def production_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def pdtype_wildcard(self, tree):
        return Wildcard()

    def production_wildcard(self, tree):
        return Wildcard()
