from io import StringIO

import numpy as np
import pandas as pd
import pytest

from pharmpy.modeling import (
    calculate_aic,
    calculate_bic,
    calculate_eta_shrinkage,
    calculate_individual_parameter_statistics,
    calculate_individual_shrinkage,
    calculate_pk_parameters_statistics,
    check_parameters_near_bounds,
    set_iiv_on_ruv,
)
from pharmpy.tools import read_modelfit_results


def test_calculate_eta_shrinkage(load_model_for_test, testdata):
    pheno = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    pe = res.parameter_estimates
    ie = res.individual_estimates
    shrinkage = calculate_eta_shrinkage(pheno, pe, ie)
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA_1'], 0.0001) == 7.2048e01 / 100
    assert pytest.approx(shrinkage['ETA_2'], 0.0001) == 2.4030e01 / 100
    shrinkage = calculate_eta_shrinkage(pheno, pe, ie, sd=True)
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA_1'], 0.0001) == 4.7130e01 / 100
    assert pytest.approx(shrinkage['ETA_2'], 0.0001) == 1.2839e01 / 100


def test_calculate_individual_shrinkage(load_model_for_test, testdata):
    pheno = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    ishr = calculate_individual_shrinkage(
        pheno,
        res.parameter_estimates,
        res.individual_estimates_covariance,
    )
    assert len(ishr) == 59
    assert pytest.approx(ishr['ETA_1'][1], 1e-15) == 0.84778949807160287


def test_calculate_individual_parameter_statistics(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'secondary_parameters' / 'pheno.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'secondary_parameters' / 'pheno.mod')
    rng = np.random.default_rng(103)
    stats = calculate_individual_parameter_statistics(
        model,
        'CL/V',
        res.parameter_estimates,
        res.covariance_matrix,
        rng=rng,
    )

    assert stats['mean'][0] == pytest.approx(0.004700589484324183)
    assert stats['variance'][0] == pytest.approx(8.086653508585209e-06)
    assert stats['stderr'][0] == pytest.approx(0.0035089729730046304, abs=1e-6)

    model = load_model_for_test(testdata / 'nonmem' / 'secondary_parameters' / 'run1.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'secondary_parameters' / 'run1.mod')
    rng = np.random.default_rng(5678)
    stats = calculate_individual_parameter_statistics(
        model,
        'CL/V',
        res.parameter_estimates,
        res.covariance_matrix,
        rng=rng,
    )
    assert stats['mean'][0] == pytest.approx(0.0049100899539843)
    assert stats['variance'][0] == pytest.approx(7.391076132098555e-07)
    assert stats['stderr'][0] == pytest.approx(0.0009425952783595735, abs=1e-6)

    covmodel = load_model_for_test(testdata / 'nonmem' / 'secondary_parameters' / 'run2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'secondary_parameters' / 'run2.mod')
    rng = np.random.default_rng(8976)
    stats = calculate_individual_parameter_statistics(
        covmodel,
        'K = CL/V',
        res.parameter_estimates,
        res.covariance_matrix,
        rng=rng,
    )
    assert stats['mean']['K', 'median'] == pytest.approx(0.004526899290470633)
    assert stats['variance']['K', 'median'] == pytest.approx(2.95125370813005e-06)
    assert stats['stderr']['K', 'median'] == pytest.approx(0.0018170955599868073, abs=1e-6)
    assert stats['mean']['K', 'p5'] == pytest.approx(0.0033049497924269385)
    assert stats['variance']['K', 'p5'] == pytest.approx(1.5730213328583985e-06)
    assert stats['stderr']['K', 'p5'] == pytest.approx(0.0013102577338191103, abs=1e-6)
    assert stats['mean']['K', 'p95'] == pytest.approx(0.014616277746303079)
    assert stats['variance']['K', 'p95'] == pytest.approx(3.0766525541426746e-05)
    assert stats['stderr']['K', 'p95'] == pytest.approx(0.006735905156223314, abs=1e-6)


def test_calculate_pk_parameters_statistics(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox1.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox1.mod')
    rng = np.random.default_rng(103)
    df = calculate_pk_parameters_statistics(
        model,
        res.parameter_estimates,
        res.covariance_matrix,
        rng=rng,
    )
    assert df['mean'].loc['t_max', 'median'] == pytest.approx(1.5999856886869577)
    assert df['variance'].loc['t_max', 'median'] == pytest.approx(0.29728565293669557)
    assert df['stderr'].loc['t_max', 'median'] == pytest.approx(0.589128711884761)
    assert df['mean'].loc['C_max_dose', 'median'] == pytest.approx(0.6305869738624813)
    assert df['variance'].loc['C_max_dose', 'median'] == pytest.approx(0.012200490462185057)
    assert df['stderr'].loc['C_max_dose', 'median'] == pytest.approx(0.11128015565024524)


def test_calc_pk_two_comp_bolus(load_model_for_test, testdata):
    # Warning: These results are based on a manually modified cov-matrix
    # Results are not verified
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox_2comp.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox_2comp.mod')
    rng = np.random.default_rng(103)
    df = calculate_pk_parameters_statistics(
        model,
        res.parameter_estimates,
        res.covariance_matrix,
        rng=rng,
    )
    # FIXME: Why doesn't random state handle this difference in stderr?
    df.drop('stderr', inplace=True, axis=1)

    correct = """parameter,covariates,mean,variance,stderr
A,median,0.003785,0.0,0.052979
B,median,0.996215,0.0,0.051654
alpha,median,0.109317,0.000037,0.940936
beta,median,24.27695,2.660843,24.759415
k_e,median,13.319584,2.67527,2.633615
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1])
    correct.index.set_names(['parameter', 'covariates'], inplace=True)
    correct.drop('stderr', inplace=True, axis=1)
    # pd.testing.assert_frame_equal(df, correct, atol=1e-4)


def test_aic(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    assert calculate_aic(model, res.ofv) == 740.8947268137307


def test_bic(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    ofv = res.ofv
    assert calculate_bic(model, ofv, type='iiv') == 739.0498017015422
    assert calculate_bic(model, ofv, type='fixed') == 756.111852398327
    assert calculate_bic(model, ofv, type='random') == 751.2824140332593
    assert calculate_bic(model, ofv) == 752.2483017062729
    model = set_iiv_on_ruv(model)
    assert calculate_bic(model, ofv) == 755.359951477165


def test_check_parameters_near_bounds(load_model_for_test, testdata):
    onePROB = testdata / 'nonmem' / 'modelfit_results' / 'onePROB'
    nearbound = load_model_for_test(onePROB / 'oneEST' / 'noSIM' / 'near_bounds.mod')
    res = read_modelfit_results(onePROB / 'oneEST' / 'noSIM' / 'near_bounds.mod')
    correct = pd.Series(
        [False, True, False, False, False, False, False, False, True, True, False],
        index=[
            'POP_CL',
            'POP_V',
            'POP_KA',
            'LAG',
            'OMEGA_1_1',
            'OMEGA_2_1',
            'IIV_CL_V',
            'IIV_KA',
            'IOV_CL',
            'IOV_KA',
            'SIGMA_1_1',
        ],
    )
    pd.testing.assert_series_equal(
        check_parameters_near_bounds(nearbound, res.parameter_estimates),
        correct,
    )
