import pytest
import sympy

from pharmpy import Model
from pharmpy.modeling import create_rv_block
from pharmpy.modeling.block_rvs import RVInputException, _extract_rv_from_block, _get_rvs
from pharmpy.random_variables import JointNormalSeparate, RandomVariables, VariabilityLevel
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

    with pytest.raises(RVInputException, match=exception_msg):
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


def test_extract_rv_from_block():
    cov = sympy.zeros(3)

    for row in range(3):
        for col in range(3):
            cov[row, col] = S(f'OMEGA({row + 1},{col + 1})')

    names = ['ETA(1)', 'ETA(2)', 'ETA(3)']
    means = [0, 0, 0]

    dist = JointNormalSeparate(names, means, cov)

    for rv in dist:
        rv.variability_level = VariabilityLevel.IIV

    rvs = RandomVariables(dist)
    _, list_of_dists = rvs.distributions_as_list()

    assert len(list_of_dists) == 1

    joined_names = ['ETA(2)', 'ETA(3)']

    rvs_new = _extract_rv_from_block(dist[0], joined_names)
    rvs_new = RandomVariables(rvs_new)

    _, list_of_dists = rvs_new.distributions_as_list()

    assert len(list_of_dists) == 2
