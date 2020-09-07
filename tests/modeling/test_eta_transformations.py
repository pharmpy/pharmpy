from sympy import exp

from pharmpy.modeling.eta_transformations import EtaTransformation
from pharmpy.symbols import real


def S(x):
    return real(x)


def test_apply():
    eta_trans = EtaTransformation.boxcox(2).assignments

    assert eta_trans[0].symbol == S('etab1')
    assert eta_trans[0].expression == (exp(S('etab1')*(S('theta1')-1)))/S('theta1')
