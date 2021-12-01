from io import StringIO

import numpy as np
import pandas as pd
import pytest

from pharmpy import Model
from pharmpy.modeling import (
    calculate_eta_shrinkage,
    calculate_individual_parameter_statistics,
    calculate_individual_shrinkage,
    calculate_pk_parameters_statistics,
    summarize_modelfit_results,
)


def test_calculate_eta_shrinkage(testdata):
    pheno = Model(testdata / 'nonmem' / 'pheno_real.mod')
    shrinkage = calculate_eta_shrinkage(pheno)
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA(1)'], 0.0001) == 7.2048e01 / 100
    assert pytest.approx(shrinkage['ETA(2)'], 0.0001) == 2.4030e01 / 100
    shrinkage = calculate_eta_shrinkage(pheno, sd=True)
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA(1)'], 0.0001) == 4.7130e01 / 100
    assert pytest.approx(shrinkage['ETA(2)'], 0.0001) == 1.2839e01 / 100


def test_calculate_individual_shrinkage(testdata):
    pheno = Model(testdata / 'nonmem' / 'pheno_real.mod')
    ishr = calculate_individual_shrinkage(pheno)
    assert len(ishr) == 59
    assert pytest.approx(ishr['ETA(1)'][1], 1e-15) == 0.84778949807160287


def test_calculate_individual_parameter_statistics(testdata):
    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'pheno.mod')
    rng = np.random.default_rng(103)
    stats = calculate_individual_parameter_statistics(model, 'CL/V', rng=rng)

    assert stats['mean'][0] == pytest.approx(0.004700589484324183)
    assert stats['variance'][0] == pytest.approx(8.086653508585209e-06)
    assert stats['stderr'][0] == pytest.approx(0.0035089729730046304, abs=1e-6)

    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run1.mod')
    rng = np.random.default_rng(5678)
    stats = calculate_individual_parameter_statistics(model, 'CL/V', rng=rng)
    assert stats['mean'][0] == pytest.approx(0.0049100899539843)
    assert stats['variance'][0] == pytest.approx(7.391076132098555e-07)
    assert stats['stderr'][0] == pytest.approx(0.0009425952783595735, abs=1e-6)

    covmodel = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run2.mod')
    rng = np.random.default_rng(8976)
    stats = calculate_individual_parameter_statistics(covmodel, 'K = CL/V', rng=rng)
    assert stats['mean']['K', 'median'] == pytest.approx(0.004525842355027405)
    assert stats['variance']['K', 'median'] == pytest.approx(2.9540381716908423e-06)
    assert stats['stderr']['K', 'median'] == pytest.approx(0.001804371451706786, abs=1e-6)
    assert stats['mean']['K', 'p5'] == pytest.approx(0.0033049497924269385)
    assert stats['variance']['K', 'p5'] == pytest.approx(1.5730213328583985e-06)
    assert stats['stderr']['K', 'p5'] == pytest.approx(0.0013102577338191103, abs=1e-6)
    assert stats['mean']['K', 'p95'] == pytest.approx(0.014625434302866478)
    assert stats['variance']['K', 'p95'] == pytest.approx(3.090546438695198e-05)
    assert stats['stderr']['K', 'p95'] == pytest.approx(0.0069971678916412716, abs=1e-6)


def test_calculate_pk_parameters_statistics(testdata):
    model = Model(testdata / 'nonmem' / 'models' / 'mox1.mod')
    rng = np.random.default_rng(103)
    df = calculate_pk_parameters_statistics(model, rng=rng)
    assert df['mean'].loc['t_max', 'median'] == pytest.approx(1.5999856886869577)
    assert df['variance'].loc['t_max', 'median'] == pytest.approx(0.29728565293669557)
    assert df['stderr'].loc['t_max', 'median'] == pytest.approx(0.589128711884761)
    assert df['mean'].loc['C_max_dose', 'median'] == pytest.approx(0.6306393134647171)
    assert df['variance'].loc['C_max_dose', 'median'] == pytest.approx(0.012194951111832641)
    assert df['stderr'].loc['C_max_dose', 'median'] == pytest.approx(0.11328030491204867)


def test_calc_pk_two_comp_bolus(testdata):
    # Warning: These results are based on a manually modified cov-matrix
    # Results are not verified
    model = Model(testdata / 'nonmem' / 'models' / 'mox_2comp.mod')
    rng = np.random.default_rng(103)
    df = calculate_pk_parameters_statistics(model, rng=rng)
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


def test_summarize_modelfit_results(testdata, pheno_path):
    pheno = Model(pheno_path)

    summary_single = summarize_modelfit_results(pheno)

    assert summary_single.loc['pheno_real']['ofv'] == 586.2760562818805
    assert summary_single['OMEGA(1,1)_estimate'].mean() == 0.0293508

    assert len(summary_single.index) == 1

    mox = Model(testdata / 'nonmem' / 'models' / 'mox1.mod')

    summary_multiple = summarize_modelfit_results([pheno, mox])

    assert summary_multiple.loc['mox1']['ofv'] == -624.5229577248352
    assert summary_multiple['OMEGA(1,1)_estimate'].mean() == 0.2236304
    assert summary_multiple['OMEGA(2,1)_estimate'].mean() == 0.395647  # One is NaN

    assert len(summary_multiple.index) == 2
    assert list(summary_multiple.index) == ['mox1', 'pheno_real']

    pheno_no_res = Model(StringIO(pheno.model_code))
    pheno_no_res.name = 'pheno_no_res'

    summary_no_res = summarize_modelfit_results([pheno, pheno_no_res])

    assert summary_no_res.loc['pheno_real']['ofv'] == 586.2760562818805
    assert np.isnan(summary_no_res.loc['pheno_no_res']['ofv'])
    assert np.all(np.isnan(summary_no_res.filter(regex='estimate$').loc['pheno_no_res']))

    pheno_multest = Model(
        testdata
        / 'nonmem'
        / 'modelfit_results'
        / 'onePROB'
        / 'multEST'
        / 'noSIM'
        / 'pheno_multEST.mod'
    )

    summary_multest = summarize_modelfit_results([pheno_multest, mox])

    assert len(summary_multest.index) == 2

    assert not summary_multest.loc['pheno_multEST']['minimization_successful']

    summary_multest_full = summarize_modelfit_results(
        [pheno_multest, mox], include_all_estimation_steps=True
    )

    assert len(summary_multest_full.index) == 3
    assert len(set(summary_multest_full.index.get_level_values('model_name'))) == 2
    assert summary_multest_full.loc['pheno_multEST', 1]['run_type'] == 'estimation'
    assert summary_multest_full.loc['pheno_multEST', 2]['run_type'] == 'evaluation'

    assert not summary_multest_full.loc['pheno_multEST', 1]['minimization_successful']

    pheno_multest_no_res = Model(StringIO(pheno_multest.model_code))
    pheno_multest_no_res.name = 'pheno_multest_no_res'

    summary_multest_full_no_res = summarize_modelfit_results(
        [pheno_multest_no_res, mox], include_all_estimation_steps=True
    )

    assert summary_multest_full_no_res.loc['mox1', 1]['ofv'] == -624.5229577248352
    assert np.isnan(summary_multest_full_no_res.loc['pheno_multest_no_res', 1]['ofv'])
    estimates = summary_multest_full_no_res.loc['pheno_multest_no_res', 2].iloc[2:]
    assert estimates.isnull().all()
    assert summary_multest_full_no_res.dtypes['minimization_successful'] == bool
