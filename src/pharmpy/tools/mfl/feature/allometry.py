from functools import partial
from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import add_allometry

from ..statement.feature.allometry import Allometry
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, Allometry):
            ref = statement.reference if statement.reference is not None else 70.0
            yield ('ALLOMETRY', statement.covariate, ref), partial(
                add_allometry, allometric_variable=statement.covariate, reference_value=ref
            )
