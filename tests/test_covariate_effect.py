from sympy import Symbol, exp

from pharmpy.covariate_effect import CovariateEffect


def S(x):
    return Symbol(x, real=True)


def test_apply():
    ce = CovariateEffect.exponential()

    ce.apply(parameter='CL', covariate='WGT', theta_name='THETA(x)',
             mean=1.52, median=1.3)

    assert ce.template.symbol == S('CLWGT')
    assert ce.template.expression == exp(S('THETA(x)') * (S('WGT')-1.3))
