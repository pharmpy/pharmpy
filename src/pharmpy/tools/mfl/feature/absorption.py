from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import (
    set_first_order_absorption,
    set_instantaneous_absorption,
    set_seq_zo_fo_absorption,
    set_zero_order_absorption,
)

from ..statement.feature.absorption import ABSORPTION_WILDCARD, Absorption
from ..statement.feature.symbols import Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, Absorption):
            modes = (
                ABSORPTION_WILDCARD if isinstance(statement.modes, Wildcard) else statement.modes
            )
            for mode in modes:
                if mode.name == 'FO':
                    yield ('ABSORPTION', mode.name), set_first_order_absorption
                elif mode.name == 'ZO':
                    yield ('ABSORPTION', mode.name), set_zero_order_absorption
                elif mode.name == 'SEQ-ZO-FO':
                    yield ('ABSORPTION', mode.name), set_seq_zo_fo_absorption
                elif mode.name == "INST":
                    yield ('ABSORPTION', mode.name), set_instantaneous_absorption
                else:
                    raise ValueError(f'Absorption {mode} not supported')
