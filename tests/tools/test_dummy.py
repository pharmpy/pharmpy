import pytest

from pharmpy.deps import numpy as np
from pharmpy.modeling import (
    get_number_of_individuals,
    remove_parameter_uncertainty_step,
    set_simulation,
)
from pharmpy.tools.external.dummy.run import (
    create_dummy_modelfit_results,
    create_dummy_simulation_results,
)


@pytest.mark.parametrize(
    'ref_value, with_uncertainty',
    [(None, False), (None, True), (-100, False)],
)
def test_create_dummy_modelfit_results(load_example_model_for_test, ref_value, with_uncertainty):
    model = load_example_model_for_test('pheno')
    if not with_uncertainty:
        model = remove_parameter_uncertainty_step(model)
    res = create_dummy_modelfit_results(model, ref_value)
    if ref_value:
        assert abs(res.ofv - ref_value) < 50
    else:
        assert abs(res.ofv) < 20
    no_of_params = len(model.parameters)
    assert len(res.parameter_estimates) == no_of_params
    assert len(res.relative_standard_errors) == no_of_params
    assert len(res.standard_errors) == no_of_params
    if not with_uncertainty:
        assert all(np.isnan(val) for val in res.relative_standard_errors.values)
    no_of_individuals = get_number_of_individuals(model)
    assert len(res.individual_ofv) == no_of_individuals
    assert len(res.individual_estimates)


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
