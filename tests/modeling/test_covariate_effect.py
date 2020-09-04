import pytest
from sympy import exp

from pharmpy import Model
from pharmpy.modeling.covariate_effect import CovariateEffect, choose_param_inits
from pharmpy.symbols import real


def S(x):
    return real(x)


@pytest.mark.parametrize('cov_eff,symbol,expression', [
    (CovariateEffect.exponential(), S('CLWGT'),
     exp(S('COVEFF1') * (S('WGT') - S('CL_MEDIAN')))),
    (CovariateEffect.power(), S('CLWGT'),
     (S('WGT')/S('CL_MEDIAN'))**S('COVEFF1')),
    (CovariateEffect.linear_continuous(), S('CLWGT'),
     1 + S('COVEFF1') * (S('WGT') - S('CL_MEDIAN')))
])
def test_apply(cov_eff, symbol, expression):
    cov_eff.apply(parameter='CL', covariate='WGT',
                  thetas={'theta': 'COVEFF1'})

    assert cov_eff.template.symbol == symbol
    assert cov_eff.template.expression == expression


def test_choose_param_inits(pheno_path):
    model = Model(pheno_path)

    lower, upper = choose_param_inits('exp', model.dataset, 'WGT')

    assert round(lower, 4) == -0.4348
    assert round(upper, 4) == 0.8696
