from functools import partial
from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import set_direct_effect

from ..statement.feature.direct_effect import DirectEffect
from ..statement.feature.symbols import Name, Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, DirectEffect):
            modes = (
                [Name('LINEAR'), Name('EMAX'), Name('SIGMOID')]
                if isinstance(statement.modes, Wildcard)
                else statement.modes
            )
            for mode in modes:
                yield ('DIRECT', mode.name), partial(set_direct_effect, expr=mode.name.lower())
