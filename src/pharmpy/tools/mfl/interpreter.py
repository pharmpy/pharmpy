from lark.visitors import Interpreter

from .statement.definition import DefinitionInterpreter
from .statement.feature.absorption import AbsorptionInterpreter
from .statement.feature.allometry import AllometryInterpreter
from .statement.feature.covariate import CovariateInterpreter
from .statement.feature.direct_effect import DirectEffectInterpreter
from .statement.feature.effect_comp import EffectCompInterpreter
from .statement.feature.elimination import EliminationInterpreter
from .statement.feature.indirect_effect import IndirectEffectInterpreter
from .statement.feature.lagtime import LagTimeInterpreter
from .statement.feature.metabolite import MetaboliteInterpreter
from .statement.feature.peripherals import PeripheralsInterpreter
from .statement.feature.transits import TransitsInterpreter
from .statement.statement import Statement


class MFLInterpreter(Interpreter):
    def interpret(self, tree) -> list[Statement]:
        return self.visit_children(tree)

    def definition(self, tree):
        return DefinitionInterpreter().interpret(tree)

    def absorption(self, tree):
        return AbsorptionInterpreter().interpret(tree)

    def elimination(self, tree):
        return EliminationInterpreter().interpret(tree)

    def transits(self, tree):
        return TransitsInterpreter().interpret(tree)

    def peripherals(self, tree):
        return PeripheralsInterpreter().interpret(tree)

    def lagtime(self, tree):
        return LagTimeInterpreter().interpret(tree)

    def covariate(self, tree):
        return CovariateInterpreter().interpret(tree)

    def allometry(self, tree):
        return AllometryInterpreter().interpret(tree)

    def direct_effect(self, tree):
        return DirectEffectInterpreter().interpret(tree)

    def effect_comp(self, tree):
        return EffectCompInterpreter().interpret(tree)

    def indirect_effect(self, tree):
        return IndirectEffectInterpreter().interpret(tree)

    def metabolite(self, tree):
        return MetaboliteInterpreter().interpret(tree)
