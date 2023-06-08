import shutil
from dataclasses import replace

import pytest

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import (
    add_iiv,
    add_population_parameter,
    create_joint_distribution,
    fix_or_unfix_parameters,
    fix_parameters,
    fix_parameters_to,
    get_omegas,
    get_sigmas,
    get_thetas,
    set_lower_bounds,
    set_upper_bounds,
    unconstrain_parameters,
    unfix_parameters,
    unfix_parameters_to,
    update_inits,
)
from pharmpy.tools import read_modelfit_results


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
    assert model.parameters['OMEGA_1_1'].upper == sympy.oo
    model = set_lower_bounds(model, {'THETA_1': -100})
    assert model.parameters['THETA_1'].lower == -100
    assert model.parameters['OMEGA_1_1'].lower == 0


def test_add_population_parameter(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_population_parameter(model, 'NEWPARAM', 23)
    assert len(model.parameters) == 4
    assert model.parameters['NEWPARAM'].init == 23


def test_update_inits_move_est(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)

    model = create_joint_distribution(model, individual_estimates=res.individual_estimates)
    model = add_iiv(model, 'S1', 'add')

    param_est = res.parameter_estimates.copy()
    param_est['IIV_CL_IIV_V'] = 0.0285  # Correlation > 0.99
    param_est['IIV_S1'] = 0.0005

    model = update_inits(model, param_est, move_est_close_to_bounds=True)

    assert model.parameters['IVCL'].init == param_est['IVCL']
    assert model.parameters['IIV_S1'].init == 0.01
    assert round(model.parameters['IIV_CL_IIV_V'].init, 6) == 0.025757


def test_update_inits_zero_fix(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    d = {name: 0 for name in model.random_variables.iiv.parameter_names}
    model = fix_parameters_to(model, d)
    res = read_modelfit_results(pheno_path)
    param_est = res.parameter_estimates.drop(index=['IVCL'])
    model = update_inits(model, param_est)
    assert model.parameters['IVCL'].init == 0
    assert model.parameters['IVCL'].fix

    model = load_model_for_test(pheno_path)
    d = {name: 0 for name in model.random_variables.iiv.parameter_names}
    model = fix_parameters_to(model, d)
    param_est = res.parameter_estimates.drop(index=['IVCL'])
    model = update_inits(model, param_est, move_est_close_to_bounds=True)
    assert model.parameters['IVCL'].init == 0
    assert model.parameters['IVCL'].fix


def test_update_inits_no_res(load_model_for_test, testdata, tmp_path):
    shutil.copy(testdata / 'nonmem/pheno.mod', tmp_path / 'run1.mod')
    shutil.copy(testdata / 'nonmem/pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        shutil.copy(testdata / 'nonmem/pheno.ext', tmp_path / 'run1.ext')
        shutil.copy(testdata / 'nonmem/pheno.lst', tmp_path / 'run1.lst')

        model = load_model_for_test('run1.mod')
        res = read_modelfit_results('run1.mod')

        modelfit_results = replace(
            res,
            parameter_estimates=pd.Series(
                np.nan, name='estimates', index=list(model.parameters.nonfixed.inits.keys())
            ),
        )

        with pytest.raises(ValueError):
            update_inits(model, modelfit_results.parameter_estimates)
