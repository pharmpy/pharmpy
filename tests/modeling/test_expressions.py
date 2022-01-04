import sympy

from pharmpy import Model
from pharmpy.modeling import get_observation_expression


def s(x):
    return sympy.Symbol(x)


def test_get_observation_expression(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_observation_expression(model)
    assert expr == s('D_EPSETA1_2') * s('EPS(1)') * (s('ETA(2)') - s('OETA2')) + s('D_ETA1') * (
        s('ETA(1)') - s('OETA1')
    ) + s('D_ETA2') * (s('ETA(2)') - s('OETA2')) + s('EPS(1)') * (
        s('D_EPS1') + s('D_EPSETA1_1') * (s('ETA(1)') - s('OETA1'))
    ) + s(
        'OPRED'
    )
