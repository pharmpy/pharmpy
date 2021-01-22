import pytest
import sympy
import sympy.stats as stats

from pharmpy import Model
from pharmpy.modeling import create_rv_block
from pharmpy.modeling.block_rvs import _choose_param_init, _get_rvs
from pharmpy.random_variables import RandomVariables, VariabilityLevel
from pharmpy.results import ModelfitResults
from pharmpy.symbols import symbol as S


@pytest.mark.parametrize(
    'rvs, exception_msg',
    [
        (['ETA(1)', 'ETA(2)'], r'.*fixed: ETA\(1\)'),
        (['ETA(3)', 'NON_EXISTENT_RV'], r'.*does not exist: NON_EXISTENT_RV'),
        (['ETA(3)', 'ETA(6)'], r'.*IOV: ETA\(6\)'),
        (['ETA(1)'], 'At least two random variables are needed'),
    ],
)
def test_incorrect_params(testdata, rvs, exception_msg):
    model = Model(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM' / 'withBayes.mod'
    )
    model.parameters
    model.random_variables

    with pytest.raises(Exception, match=exception_msg):
        create_rv_block(model, rvs)


def test_get_rvs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_block.mod')
    rvs = _get_rvs(model, None)
    assert rvs[0].name == 'ETA(1)'

    model.parameters.fix = {'OMEGA(1,1)': True}
    rvs = _get_rvs(model, None)
    assert rvs[0].name == 'ETA(2)'

    model = Model(testdata / 'nonmem' / 'pheno_block.mod')
    model.random_variables['ETA(1)'].variability_level = VariabilityLevel.IOV
    rvs = _get_rvs(model, None)
    assert rvs[0].name == 'ETA(2)'


def test_choose_param_init(pheno_path, testdata):
    model = Model(pheno_path)
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = RandomVariables(model.random_variables.etas)
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0118179

    model = Model(pheno_path)
    model.source.path = testdata  # Path where there is no .ext-file
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0031045

    model = Model(pheno_path)

    omega1 = S('OMEGA(3,3)')
    x = stats.Normal('ETA(3)', 0, sympy.sqrt(omega1))
    x.variability_level = VariabilityLevel.IIV
    rvs.add(x)

    ie = model.modelfit_results.individual_estimates
    ie['ETA(3)'] = ie['ETA(1)']
    model.modelfit_results = ModelfitResults(individual_estimates=ie)

    init = _choose_param_init(model, rvs, params)

    assert init == 0.0118179
