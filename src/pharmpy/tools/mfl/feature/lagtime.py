from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import add_lag_time

from ..statement.feature.lagtime import LagTime
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, LagTime):
            yield ('LAGTIME',), add_lag_time
