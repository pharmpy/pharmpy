import pytest

from pharmpy.deps import numpy as np
from pharmpy.modeling import set_simulation
from pharmpy.tools.external.dummy.run import create_dummy_simulation_results


@pytest.mark.parametrize(
    'n',
    [1, 3],
)
def test_create_dummy_simulation_results(load_example_model_for_test, n):
    model = load_example_model_for_test('pheno')
    model = set_simulation(model, n)
    res = create_dummy_simulation_results(model)
    dv = model.dataset['DV'].values
    for i in range(1, n + 1):
        dv_sim = res.table.loc[i, 'DV'].values
        assert len(dv) == len(dv_sim)
        assert not np.array_equal(dv, dv_sim)
