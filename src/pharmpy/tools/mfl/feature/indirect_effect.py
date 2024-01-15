from functools import partial
from itertools import product
from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import add_indirect_effect

from ..statement.feature.indirect_effect import (
    INDIRECT_EFFECT_MODES_WILDCARD,
    INDIRECT_EFFECT_PRODUCTION_WILDCARD,
    IndirectEffect,
)
from ..statement.feature.symbols import Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, IndirectEffect):
            modes = (
                INDIRECT_EFFECT_MODES_WILDCARD
                if isinstance(statement.modes, Wildcard)
                else statement.modes
            )
            production = (
                INDIRECT_EFFECT_PRODUCTION_WILDCARD
                if isinstance(statement.production, Wildcard)
                else statement.production
            )

            params = list(product(modes, production))
            params = [(mode.name, production.name) for mode, production in params]

            for param in params:
                if param[1] == 'PRODUCTION':
                    yield ('INDIRECT', *param), partial(
                        add_indirect_effect, expr=param[0].lower(), prod=True
                    )
                elif param[1] == 'DEGRADATION':
                    yield ('INDIRECT', *param), partial(
                        add_indirect_effect, expr=param[0].lower(), prod=False
                    )
