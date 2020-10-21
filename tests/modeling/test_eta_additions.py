from operator import add, mul

import pytest
import sympy

from pharmpy.modeling.eta_additions import EtaAddition
from pharmpy.symbols import symbol as S


@pytest.mark.parametrize(
    'addition,expression',
    [
        (EtaAddition.exponential(add), S('CL') + sympy.exp(S('eta_new'))),
        (EtaAddition.exponential(mul), S('CL') * sympy.exp(S('eta_new'))),
    ],
)
def test_apply(addition, expression):
    addition.apply(original='CL', eta='eta_new')

    assert addition.template == expression
