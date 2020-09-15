import pytest
from sympy import exp

import pharmpy.symbols
from pharmpy import Model
from pharmpy.modeling.covariate_effect import CovariateEffect, _choose_param_inits


def S(x):
    return pharmpy.symbols.symbol(x)


@pytest.mark.parametrize('cov_eff,symbol,expression', [
    (CovariateEffect.exponential(), S('CLWGT'),
     exp(S('COVEFF1') * (S('WGT') - S('WGT_MEDIAN')))),
    (CovariateEffect.power(), S('CLWGT'),
     (S('WGT')/S('WGT_MEDIAN'))**S('COVEFF1')),
    (CovariateEffect.linear(), S('CLWGT'),
     1 + S('COVEFF1') * (S('WGT') - S('WGT_MEDIAN')))
])
def test_apply(cov_eff, symbol, expression):
    cov_eff.apply(parameter='CL', covariate='WGT',
                  thetas={'theta': 'COVEFF1'})

    assert cov_eff.template.symbol == symbol
    assert cov_eff.template.expression == expression


def test_choose_param_inits(pheno_path):
    model = Model(pheno_path)

    init, lower, upper = _choose_param_inits('exp', model.dataset, 'WGT')

    assert init == 0.001
    assert round(lower, 4) == -0.8696
    assert round(upper, 4) == 0.8696
