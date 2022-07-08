from functools import partial
from typing import Iterable

from pharmpy.model import Model
from pharmpy.modeling import set_peripheral_compartments

from ..statement.feature.peripherals import Peripherals
from ..statement.statement import Statement
from .feature import Feature


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, Peripherals):
            for count in statement.counts:
                yield ('PERIPHERALS', count), partial(set_peripheral_compartments, n=count)
