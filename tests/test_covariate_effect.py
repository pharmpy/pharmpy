import pytest
from sympy import Symbol, exp

from pharmpy.model_transformation.covariate_effect import CovariateEffect


def S(x):
    return Symbol(x, real=True)


@pytest.mark.parametrize('cov_eff,symbol,expression', [
    (CovariateEffect.exponential(), S('CLWGT'),
     exp(S('COVEFF1') * (S('WGT') - S('CL_MEDIAN')))),
    (CovariateEffect.power(), S('CLWGT'),
     (S('WGT')/S('CL_MEDIAN'))**S('COVEFF1')),
    (CovariateEffect.linear_continuous(), S('CLWGT'),
     1 + S('COVEFF1') * (S('WGT') - S('CL_MEDIAN')))
])
def test_apply(cov_eff, symbol, expression):
    cov_eff.apply(parameter='CL', covariate='WGT', theta_name='COVEFF1')

    assert cov_eff.template.symbol == symbol
    assert cov_eff.template.expression == expression
