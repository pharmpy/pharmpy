from typing import List

from lark.visitors import Interpreter

from .statement.definition import DefinitionInterpreter
from .statement.feature.absorption import AbsorptionInterpreter
from .statement.feature.covariate import CovariateInterpreter
from .statement.feature.elimination import EliminationInterpreter
from .statement.feature.lagtime import LagTimeInterpreter
from .statement.feature.peripherals import PeripheralsInterpreter
from .statement.feature.transits import TransitsInterpreter
from .statement.statement import Statement


class MFLInterpreter(Interpreter):
    def interpret(self, tree) -> List[Statement]:
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
