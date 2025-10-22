import shutil
from dataclasses import replace

import pytest

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import (
    add_iiv,
    add_iov,
    add_population_parameter,
    create_joint_distribution,
    fix_or_unfix_parameters,
    fix_parameters,
    fix_parameters_to,
    get_omegas,
    get_sigmas,
    get_thetas,
    map_eta_parameters,
    replace_fixed_thetas,
    set_initial_estimates,
    set_lower_bounds,
    set_upper_bounds,
    unconstrain_parameters,
    unfix_parameters,
    unfix_parameters_to,
)
from pharmpy.tools.external.results import parse_modelfit_results


def test_get_thetas(pheno):
    thetas = get_thetas(pheno)
    assert len(thetas) == 3
    assert thetas['PTVCL'].init == 0.00469307


def test_get_omegas(pheno):
    omegas = get_omegas(pheno)
    assert len(omegas) == 2
    assert omegas['IVCL'].init == 0.0309626


def test_get_sigmas(pheno):
    sigmas = get_sigmas(pheno)
    assert len(sigmas) == 1
    assert sigmas['SIGMA_1_1'].init == 0.013241


def test_fix_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA_1'].fix
    model = fix_parameters(model, ['THETA_1'])
    assert model.parameters['THETA_1'].fix

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA_1'].fix
    model = fix_parameters(model, 'THETA_1')
    assert model.parameters['THETA_1'].fix

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert 'x' not in model.parameters.names

    with pytest.raises(ValueError):
        fix_parameters(model, ['x'])

    params_before = model.parameters
    model = fix_parameters(model, ['x'], strict=False)
    assert params_before == model.parameters


def test_unfix_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_parameters(model, ['THETA_1'])
    assert model.parameters['THETA_1'].fix
    model = unfix_parameters(model, ['THETA_1'])
    assert not model.parameters['THETA_1'].fix

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_parameters(model, 'THETA_1')
    assert model.parameters['THETA_1'].fix
    model = unfix_parameters(model, 'THETA_1')
    assert not model.parameters['THETA_1'].fix


def test_fix_or_unfix_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_or_unfix_parameters(model, {'THETA_1': True})
    assert model.parameters['THETA_1'].fix
    model = fix_or_unfix_parameters(model, {'THETA_1': False})
    assert not model.parameters['THETA_1'].fix


def test_unconstrain_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    fix_or_unfix_parameters(model, {'THETA_1': True})
    unconstrain_parameters(model, ['THETA_1'])
    assert not model.parameters['THETA_1'].fix
    fix_or_unfix_parameters(model, {'THETA_1': True})
    unconstrain_parameters(model, 'THETA_1')
    assert not model.parameters['THETA_1'].fix


def test_fix_parameters_to(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_parameters_to(model, {'THETA_1': 0})
    assert model.parameters['THETA_1'].fix
    assert model.parameters['THETA_1'].init == 0

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_parameters_to(model, {'THETA_1': 0, 'OMEGA_1_1': 0})
    assert model.parameters['THETA_1'].fix
    assert model.parameters['THETA_1'].init == 0
    assert model.parameters['THETA_1'].fix
    assert model.parameters['OMEGA_1_1'].init == 0

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_parameters_to(model, {'THETA_1': 0, 'OMEGA_1_1': 1})
    assert model.parameters['THETA_1'].init == 0
    assert model.parameters['OMEGA_1_1'].init == 1


def test_unfix_parameters_to(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_parameters(model, ['THETA_1'])
    assert model.parameters['THETA_1'].fix
    model = unfix_parameters_to(model, {'THETA_1': 0})
    assert not model.parameters['THETA_1'].fix
    assert model.parameters['THETA_1'].init == 0

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = fix_parameters(model, ['THETA_1', 'OMEGA_1_1'])
    assert model.parameters['THETA_1'].fix
    assert model.parameters['OMEGA_1_1'].fix
    model = unfix_parameters_to(model, {'THETA_1': 0, 'OMEGA_1_1': 0})
    assert not model.parameters['THETA_1'].fix
    assert not model.parameters['OMEGA_1_1'].fix
    assert model.parameters['THETA_1'].init == 0
    assert model.parameters['OMEGA_1_1'].init == 0


def test_set_bounds(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_upper_bounds(model, {'THETA_1': 100})
    assert model.parameters['THETA_1'].upper == 100
    assert model.parameters['OMEGA_1_1'].upper == float("inf")
    model = set_lower_bounds(model, {'THETA_1': -100})
    assert model.parameters['THETA_1'].lower == -100
    assert model.parameters['OMEGA_1_1'].lower == 0


def test_add_population_parameter(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_population_parameter(model, 'NEWPARAM', 23)
    assert len(model.parameters) == 4
    assert model.parameters['NEWPARAM'].init == 23


def test_set_initial_estimates_move_est(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = parse_modelfit_results(model, pheno_path)

    model = create_joint_distribution(model, individual_estimates=res.individual_estimates)
    model = add_iiv(model, 'S1', 'add')

    param_est = res.parameter_estimates.copy()
    param_est['IIV_CL_IIV_V'] = 0.0285  # Correlation > 0.99
    param_est['IIV_S1'] = 0.0005
    model1 = set_initial_estimates(model, param_est, move_est_close_to_bounds=True)

    assert model1.parameters['IVCL'].init == param_est['IVCL']
    assert model1.parameters['IIV_S1'].init == 0.01
    assert round(model1.parameters['IIV_CL_IIV_V'].init, 6) == 0.025757

    param_est['IIV_CL_IIV_V'] = -0.3  # Correlation < -0.99

    model2 = set_initial_estimates(model, param_est, move_est_close_to_bounds=True)
    assert model2.parameters['IVCL'].init == param_est['IVCL']
    assert round(model2.parameters['IIV_CL_IIV_V'].init, 6) == -0.025757

    model3 = set_upper_bounds(model, {'THETA_3': 1})

    param_est['THETA_3'] = -0.99

    model4 = set_initial_estimates(model3, param_est, move_est_close_to_bounds=True)
    assert model4.parameters['IVCL'].init == param_est['IVCL']
    assert model4.parameters['THETA_3'].init == -0.9405

    param_est['THETA_3'] = 0.99

    model4 = set_initial_estimates(model3, param_est, move_est_close_to_bounds=True)
    assert model4.parameters['IVCL'].init == param_est['IVCL']
    assert model4.parameters['THETA_3'].init == 0.95

    model5 = set_upper_bounds(model3, {'THETA_3': 0.1})
    model5 = set_lower_bounds(model5, {'THETA_3': 0.09})
    param_est['THETA_3'] = 0.099
    model5 = set_initial_estimates(model5, param_est, move_est_close_to_bounds=True)
    assert model5.parameters['THETA_3'].init == 0.095

    model6 = set_initial_estimates(model, {'THETA_3': float("nan")}, strict=False)
    assert model6.parameters['THETA_3'] == model.parameters['THETA_3']


def test_set_initial_estimates_zero_fix(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = parse_modelfit_results(model, pheno_path)
    d = {name: 0 for name in model.random_variables.iiv.parameter_names}
    model = fix_parameters_to(model, d)
    param_est = res.parameter_estimates.drop(index=['IVCL'])
    model = set_initial_estimates(model, param_est)
    assert model.parameters['IVCL'].init == 0
    assert model.parameters['IVCL'].fix

    model = load_model_for_test(pheno_path)
    d = {name: 0 for name in model.random_variables.iiv.parameter_names}
    model = fix_parameters_to(model, d)
    param_est = res.parameter_estimates.drop(index=['IVCL'])
    model = set_initial_estimates(model, param_est, move_est_close_to_bounds=True)
    assert model.parameters['IVCL'].init == 0
    assert model.parameters['IVCL'].fix


def test_set_initial_estimates_no_res(load_model_for_test, testdata, tmp_path):
    shutil.copy(testdata / 'nonmem/pheno.mod', tmp_path / 'run1.mod')
    shutil.copy(testdata / 'nonmem/pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        shutil.copy(testdata / 'nonmem/pheno.ext', tmp_path / 'run1.ext')
        shutil.copy(testdata / 'nonmem/pheno.lst', tmp_path / 'run1.lst')

        model = load_model_for_test('run1.mod')
        res = parse_modelfit_results(model, 'run1.mod')

        modelfit_results = replace(
            res,
            parameter_estimates=pd.Series(
                np.nan, name='estimates', index=list(model.parameters.nonfixed.inits.keys())
            ),
        )

        with pytest.raises(ValueError):
            set_initial_estimates(model, modelfit_results.parameter_estimates)


def test_set_initial_estimates_subset_parameters_w_correlation(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = parse_modelfit_results(model, pheno_path)

    model = create_joint_distribution(model, individual_estimates=res.individual_estimates)
    model = add_iiv(model, 'S1', 'add')

    param_est = {}
    param_est['IIV_CL_IIV_V'] = 0.0285  # Correlation > 0.99
    param_est['IIV_S1'] = 0.5

    updated_model = set_initial_estimates(model, param_est, move_est_close_to_bounds=True)

    assert model.parameters['IVCL'].init == updated_model.parameters['IVCL'].init
    assert model.parameters['IIV_S1'].init == 0.09
    assert updated_model.parameters['IIV_S1'].init == 0.5
    assert updated_model.parameters['IIV_CL_IIV_V'].init == 0.0285


def test_replace_fixed_thetas(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = fix_parameters(model, ["POP_CL"])
    m2 = replace_fixed_thetas(model)
    assert len(m2.parameters) == len(model.parameters) - 1
    assert 'POP_CL' not in m2.parameters
    assert len(m2.statements) == len(model.statements) + 1
    assert m2.statements[0].symbol.name == 'POP_CL'


@pytest.mark.parametrize(
    'transform,keys,values,level,correct',
    [
        (lambda x: x, 'parameters', 'etas', 'iiv', {'CL': ['ETA_CL'], 'VC': ['ETA_VC']}),
        (lambda x: x, 'etas', 'parameters', 'iiv', {'ETA_CL': ['CL'], 'ETA_VC': ['VC']}),
        (lambda x: x, 'parameters', 'omegas', 'iiv', {'CL': ['IIV_CL'], 'VC': ['IIV_VC']}),
        (lambda x: x, 'omegas', 'parameters', 'iiv', {'IIV_CL': ['CL'], 'IIV_VC': ['VC']}),
        (lambda x: x, 'omegas', 'etas', 'iiv', {'IIV_CL': ['ETA_CL'], 'IIV_VC': ['ETA_VC']}),
        (lambda x: x, 'etas', 'omegas', 'iiv', {'ETA_CL': ['IIV_CL'], 'ETA_VC': ['IIV_VC']}),
        (lambda x: x, 'parameters', 'etas', 'iov', {}),
        (lambda x: x, 'etas', 'parameters', 'iov', {}),
        (lambda x: x, 'parameters', 'omegas', 'iov', {}),
        (lambda x: x, 'omegas', 'parameters', 'iov', {}),
        (lambda x: x, 'omegas', 'etas', 'iov', {}),
        (lambda x: x, 'etas', 'omegas', 'iov', {}),
        (
            lambda x: add_iov(x, "FA1", "CL"),
            'etas',
            'omegas',
            'iov',
            {'ETA_IOV_1_1': ['OMEGA_IOV_1'], 'ETA_IOV_1_2': ['OMEGA_IOV_1']},
        ),
        (
            lambda x: add_iov(x, "FA1", "CL"),
            'omegas',
            'etas',
            'iov',
            {'OMEGA_IOV_1': ['ETA_IOV_1_1', 'ETA_IOV_1_2']},
        ),
        (
            lambda x: add_iov(x, "FA1", "CL"),
            'parameters',
            'etas',
            'iov',
            {'CL': ['ETA_IOV_1_1', 'ETA_IOV_1_2']},
        ),
        (
            lambda x: add_iov(x, "FA1", "CL"),
            'parameters',
            'etas',
            'iov',
            {'CL': ['ETA_IOV_1_1', 'ETA_IOV_1_2']},
        ),
    ],
)
def test_map_eta_parameters(load_example_model_for_test, transform, keys, values, level, correct):
    model = load_example_model_for_test("pheno")
    model = transform(model)
    d = map_eta_parameters(model, keys, values, level=level)
    assert d == correct
