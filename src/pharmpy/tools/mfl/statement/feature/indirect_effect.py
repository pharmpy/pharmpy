from dataclasses import dataclass
from typing import Literal, Tuple, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard


@dataclass(frozen=True)
class IndirectEffect(ModelFeature):
    modes: Union[Tuple[Name[Literal['LINEAR', 'EMAX', 'SIGMOID']], ...], Wildcard]
    production: Union[Tuple[Name[Literal['DEGRADATION', 'PRODUCTION']], ...], Wildcard]

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            modes = self._wildcard("modes")
        else:
            modes = self.modes

        if isinstance(self.production, Wildcard):
            production = self._wildcard("production")
        else:
            production = self.production

        return IndirectEffect(modes, production)

    def _wildcard(self, attr):
        if attr == "modes":
            return tuple([Name(x) for x in ['LINEAR', 'EMAX', 'SIGMOID']])
        elif attr == "production":
            return tuple([Name(x) for x in ['DEGRADATION', 'PRODUCTION']])


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
