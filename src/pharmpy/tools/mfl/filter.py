from pharmpy.tools.mfl.statement.definition import Let
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.direct_effect import DirectEffect
from pharmpy.tools.mfl.statement.feature.effect_comp import EffectComp
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.indirect_effect import IndirectEffect
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.metabolite import Metabolite
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.transits import Transits

from .statement.feature.symbols import Name

MODELSEARCH_STATEMENT_TYPES = (
    Absorption,
    Elimination,
    LagTime,
    Transits,
)

COVSEARCH_STATEMENT_TYPES = (
    Let,
    Covariate,
)


STRUCTSEARCH_STATEMENT_TYPES = (DirectEffect, EffectComp, IndirectEffect, Metabolite)


def mfl_filtering(statements, tool_name):
    assert tool_name in ["structsearch", "modelsearch"]

    if tool_name == "modelsearch":
        statement_types = MODELSEARCH_STATEMENT_TYPES
        peripheral_name = Name('DRUG')
    elif tool_name == "structsearch":
        statement_types = STRUCTSEARCH_STATEMENT_TYPES
        peripheral_name = Name('MET')

    filtered_statements = []
    for statement in statements:
        if isinstance(statement, statement_types):
            filtered_statements.append(statement)
        elif isinstance(statement, Peripherals) and peripheral_name in statement.modes:
            filtered_statements.append(Peripherals(statement.counts, (peripheral_name,)))
    return tuple(filtered_statements)
