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
                for mode in statement.modes:
                    if mode.name == "DRUG":
                        yield ('PERIPHERALS', count), partial(set_peripheral_compartments, n=count)
                    elif mode.name == "MET":
                        # TODO: Update how we find name of metabolite compartment
                        name = "METABOLITE"
                        yield ('PERIPHERALS', count), partial(
                            set_peripheral_compartments, n=count, name=name
                        )
                    else:
                        raise ValueError(
                            f"Unknown mode ({mode.name}) for peripheral compartment specified"
                        )
