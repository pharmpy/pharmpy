from functools import partial
from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import add_metabolite

from ..statement.feature.metabolite import METABOLITE_WILDCARD, Metabolite
from ..statement.feature.symbols import Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, Metabolite):
            modes = (
                METABOLITE_WILDCARD if isinstance(statement.modes, Wildcard) else statement.modes
            )
            for mode in modes:
                yield ('METABOLITE', mode.name), partial(
                    add_metabolite, presystemic=True if mode.name == "PSC" else False
                )
