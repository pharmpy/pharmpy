import pytest
from sympy import exp

from pharmpy.modeling.eta_transformations import EtaTransformation
from pharmpy.symbols import real


def S(x):
    return real(x)


@pytest.mark.parametrize('eta_trans,symbol,expression', [
    (EtaTransformation.boxcox(2), S('ETAB(2)'),
     ((exp(S('ETA(2)') ** (S('COVEFF2') - 1))) / S('COVEFF2'))),
])
def test_apply(eta_trans, symbol, expression):
    etas = {'eta1': real('ETA(1)'),
            'eta2': real('ETA(2)'),
            'etab1': real('ETAB(1)'),
            'etab2': real('ETAB(2)')}

    thetas = {'theta1': 'COVEFF1',
              'theta2': 'COVEFF2'}

    eta_trans.apply(etas, thetas)
    assert eta_trans.assignments[1].symbol == symbol
    assert eta_trans.assignments[1].expression == expression
