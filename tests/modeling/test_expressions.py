import sympy

from pharmpy import Model
from pharmpy.modeling import (
    calculate_epsilon_gradient_expression,
    calculate_eta_gradient_expression,
    get_individual_prediction_expression,
    get_observation_expression,
    get_population_prediction_expression,
)


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


def test_get_individual_prediction_expression(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_individual_prediction_expression(model)
    assert expr == s('D_ETA1') * (s('ETA(1)') - s('OETA1')) + s('D_ETA2') * (
        s('ETA(2)') - s('OETA2')
    ) + s('OPRED')


def test_get_population_prediction_expression(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_population_prediction_expression(model)
    assert expr == -s('D_ETA1') * s('OETA1') - s('D_ETA2') * s('OETA2') + s('OPRED')


def test_calculate_eta_gradient_expression(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = calculate_eta_gradient_expression(model)
    assert expr == [s('D_ETA1'), s('D_ETA2')]


def test_calculate_epsilon_gradient_expression(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = calculate_epsilon_gradient_expression(model)
    assert expr == [
        s('D_EPS1')
        + s('D_EPSETA1_1') * (s('ETA(1)') - s('OETA1'))
        + s('D_EPSETA1_2') * (s('ETA(2)') - s('OETA2'))
    ]
