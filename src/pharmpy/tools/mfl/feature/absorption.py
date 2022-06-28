from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import (
    set_first_order_absorption,
    set_seq_zo_fo_absorption,
    set_zero_order_absorption,
)

from ..statement.feature.absorption import Absorption
from ..statement.feature.symbols import Name, Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, Absorption):
            modes = (
                [Name('FO'), Name('ZO'), Name('SEQ-ZO-FO')]
                if isinstance(statement.modes, Wildcard)
                else statement.modes
            )
            for mode in modes:
                if mode.name == 'FO':
                    yield ('ABSORPTION', mode.name), set_first_order_absorption
                elif mode.name == 'ZO':
                    yield ('ABSORPTION', mode.name), set_zero_order_absorption
                elif mode.name == 'SEQ-ZO-FO':
                    yield ('ABSORPTION', mode.name), set_seq_zo_fo_absorption
                else:
                    raise ValueError(f'Absorption {mode} not supported')
