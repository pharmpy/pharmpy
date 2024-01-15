from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import add_lag_time, remove_lag_time

from ..statement.feature.lagtime import LAGTIME_WILDCARD, LagTime
from ..statement.feature.symbols import Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, LagTime):
            modes = LAGTIME_WILDCARD if isinstance(statement.modes, Wildcard) else statement.modes
            for mode in modes:
                if mode.name == "ON":
                    yield ('LAGTIME', mode.name), add_lag_time
                elif mode.name == "OFF":
                    yield ('LAGTIME', mode.name), remove_lag_time
                else:
                    raise ValueError(f'Lagtime {mode} not supported')
