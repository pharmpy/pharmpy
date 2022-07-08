from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import (
    set_first_order_elimination,
    set_michaelis_menten_elimination,
    set_mixed_mm_fo_elimination,
    set_zero_order_elimination,
)

from ..statement.feature.elimination import Elimination
from ..statement.feature.symbols import Name, Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, Elimination):
            modes = (
                [Name('FO'), Name('ZO'), Name('MM'), Name('MIX-FO-MM')]
                if isinstance(statement.modes, Wildcard)
                else statement.modes
            )
            for mode in modes:
                if mode.name == 'FO':
                    yield ('ELIMINATION', mode.name), set_first_order_elimination
                elif mode.name == 'ZO':
                    yield ('ELIMINATION', mode.name), set_zero_order_elimination
                elif mode.name == 'MM':
                    yield ('ELIMINATION', mode.name), set_michaelis_menten_elimination
                elif mode.name == 'MIX-FO-MM':
                    yield ('ELIMINATION', mode.name), set_mixed_mm_fo_elimination
                else:
                    raise ValueError(f'Elimination {mode} not supported')
