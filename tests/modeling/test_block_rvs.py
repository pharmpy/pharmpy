import pytest

from pharmpy import Model
from pharmpy.modeling import create_rv_block
from pharmpy.modeling.block_rvs import RVInputException


@pytest.mark.parametrize(
    'rvs, exception_msg',
    [
        (['ETA(1)', 'ETA(2)'], r'.*fixed: ETA\(1\)'),
        (['ETA(3)', 'NON_EXISTENT_RV'], r'.*does not exist: NON_EXISTENT_RV'),
        (['ETA(3)', 'ETA(6)'], r'.*IOV: ETA\(6\)'),
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
