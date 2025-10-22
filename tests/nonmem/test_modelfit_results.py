import re
import shutil

import pytest

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Parameter, Parameters
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.external.nonmem.results import parse_modelfit_results, simfit_results
from pharmpy.workflows.results import read_results


def test_ofv(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    assert res.ofv == 586.27605628188053


def test_failed_ofv(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'errors' / 'failed_run.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert np.isnan(res.ofv)
    assert res.parameter_estimates.isnull().all()

    path = testdata / 'nonmem' / 'errors' / 'run_interrupted.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert np.isnan(res.ofv)
    assert res.parameter_estimates.isnull().all()


def test_special_models_bayes(testdata, load_model_for_test):
    onePROB = testdata / 'nonmem' / 'modelfit_results' / 'onePROB'
    path = onePROB / 'multEST' / 'noSIM' / 'withBayes.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert pytest.approx(res.standard_errors['POP_CL'], 1e-13) == 2.51942e00
    succ1 = res.minimization_successful_iterations.iloc[0]
    succ2 = res.minimization_successful_iterations.iloc[1]
    assert succ1 is not None and not succ1
    assert succ2 is not None and not succ2


def test_special_models_eval0(testdata, load_model_for_test):
    onePROB = testdata / 'nonmem' / 'modelfit_results' / 'onePROB'
    path = onePROB / 'oneEST' / 'noSIM' / 'maxeval0.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert res.minimization_successful is None


def test_special_models_eval3(testdata, load_model_for_test):
    onePROB = testdata / 'nonmem' / 'modelfit_results' / 'onePROB'
    path = onePROB / 'oneEST' / 'noSIM' / 'maxeval3.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert res.minimization_successful is False


def test_covariance_pheno(pheno_path, pheno):
    res = parse_modelfit_results(pheno, pheno_path)
    cov = res.covariance_matrix
    assert len(cov) == 6
    assert pytest.approx(cov.loc['PTVCL', 'PTVCL'], 1e-13) == 4.41151e-08
    assert pytest.approx(cov.loc['IVV', 'PTVV'], 1e-13) == 7.17184e-05


def test_covariance_pheno5(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'models' / 'pheno5.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert res.covstep_successful is True


def test_gradients(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    gradients = res.gradients
    gradients_iterations = res.gradients_iterations
    assert len(gradients) == 5
    assert not gradients_iterations.empty
    assert gradients_iterations.columns.to_list()[1::] == model.parameters.names
    assert res.covstep_successful is None

    # test correct order of parameters
    tvcl = Parameter.create('TVCL', 0.1)
    tvv = Parameter.create('TVV', 0.1)
    ivcl = Parameter.create('IVCL', 0.1)
    ivv = Parameter.create('IVV', 0.1)
    sigma = Parameter.create('SIGMA_1_1', 0.1)
    params = Parameters.create([sigma, ivv, tvcl, ivcl, tvv])
    params_before = model.parameters.names
    model = model.replace(parameters=params)
    res = parse_modelfit_results(model, path=testdata / 'nonmem/pheno.mod')
    assert res.gradients_iterations.columns.to_list()[1::] == params_before


def test_information(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    m = res.precision_matrix
    assert len(m) == 6
    assert pytest.approx(m.loc['PTVCL', 'PTVCL'], 1e-13) == 2.99556e07
    assert pytest.approx(m.loc['IVV', 'PTVV'], 1e-13) == -2.80082e03


def test_correlation(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    corr = res.correlation_matrix
    assert len(corr) == 6
    assert corr.loc['PTVCL', 'PTVV'] == 0.00709865
    assert pytest.approx(corr.loc['IVV', 'PTVV'], 1e-13) == 3.56662e-01


def test_standard_errors(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    ses = res.standard_errors
    assert len(ses) == 6
    assert pytest.approx(ses['PTVCL'], 1e-13) == 2.10036e-04

    ses_sd = res.standard_errors_sdcorr
    correct = pd.Series(
        {
            'PTVCL': 0.000210036,
            'PTVV': 0.0268952,
            'THETA_3': 0.0837623,
            'IVCL': 0.0391526,
            'IVV': 0.0223779,
            'SIGMA_1_1': 0.00990444,
        }
    )
    correct.name = 'SE'
    pd.testing.assert_series_equal(ses_sd, correct)


def test_individual_ofv(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    iofv = res.individual_ofv
    assert len(iofv) == 59
    assert pytest.approx(iofv[1], 1e-15) == 5.9473520242962552
    assert pytest.approx(iofv[57], 1e-15) == 5.6639479151436394


def test_individual_estimates_basic(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    ie = res.individual_estimates
    assert len(ie) == 59
    assert pytest.approx(ie['ETA_1'][1], 1e-15) == -0.0438608
    assert pytest.approx(ie['ETA_2'][1], 1e-15) == 0.00543031
    assert pytest.approx(ie['ETA_1'][28], 1e-15) == 7.75957e-04
    assert pytest.approx(ie['ETA_2'][28], 1e-15) == 8.32311e-02


def test_individual_estimates_covariance(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    cov = res.individual_estimates_covariance
    assert len(cov) == 59
    names = ['ETA_1', 'ETA_2']
    correct = pd.DataFrame(
        [[2.48833e-02, -2.99920e-03], [-2.99920e-03, 7.15713e-03]], index=names, columns=names
    )
    pd.testing.assert_frame_equal(cov[1], correct)
    correct2 = pd.DataFrame(
        [[2.93487e-02, -1.95747e-04], [-1.95747e-04, 8.94118e-03]], index=names, columns=names
    )
    pd.testing.assert_frame_equal(cov[43], correct2)


def test_parameter_estimates(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    pe = res.parameter_estimates
    assert len(pe) == 6
    assert pe['PTVCL'] == 4.69555e-3
    assert pe['IVV'] == 2.7906e-2


def test_parameter_estimates_ext_missing_fix(load_model_for_test, pheno_path, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'errors' / 'run_interrupted.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'errors' / 'run_interrupted.mod')
    assert len(res.parameter_estimates.index.values) == len(model.parameters)


def test_simfit(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'modelfit_results' / 'simfit' / 'sim-1.mod'
    model = load_model_for_test(path)
    results = simfit_results(model, path)
    assert len(results) == 3
    assert results[1].ofv == 565.84904364342981
    assert results[2].ofv == 570.73440114145342


def test_residuals(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'pheno_real.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    df = res.residuals
    assert len(df) == 155
    assert list(df.columns) == ['RES', 'CWRES']
    assert df.loc[1, 'RES'] == -0.67071
    assert df.loc[1, 'CWRES'] == -0.401100


def test_predictions(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'pheno_real.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    df = res.predictions
    assert len(df) == 744
    assert set(df.columns) == {'IPRED', 'PRED'}
    assert df.loc[0, 'PRED'] == 18.143


def test_runtime_total(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'pheno_real.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)

    runtime = res.runtime_total
    assert runtime == 4


@pytest.mark.parametrize(
    'starttime, endtime, runtime_ref',
    [
        (
            'Sat Sep 8 10:57:25 CEST 2018',
            'Sat Sep 8 10:57:29 CEST 2018',
            4,
        ),
        (
            '08/09/2018\n10:57:25',
            '08/09/2018\n10:57:29',
            4,
        ),
        (
            '2018-09-08\n10:57:25',
            '2018-09-08\n10:57:29',
            4,
        ),
        (
            '2018-09-08\n10:57',
            '2018-09-08\n10:57',
            0,
        ),
        (
            '2018-09-08\n10:57',
            '2018-09-08\n10:58',
            60,
        ),
    ],
)
def test_runtime_different_formats(
    load_model_for_test, testdata, starttime, endtime, runtime_ref, tmp_path
):
    with open(testdata / 'nonmem' / 'pheno_real.lst', encoding='utf-8') as lst_file:
        lst_file_str = lst_file.read()

        repl_dict = {
            'start': ('lör  8 sep 2018 10:57:25 CEST', starttime),
            'end': ('lör  8 sep 2018 10:57:29 CEST', endtime),
        }

        lst_file_repl = re.sub(
            repl_dict['start'][0],
            repl_dict['start'][1],
            lst_file_str,
        )
        lst_file_repl = re.sub(
            repl_dict['end'][0],
            repl_dict['end'][1],
            lst_file_repl,
        )
        assert repl_dict['start'][0] not in lst_file_repl
        assert repl_dict['end'][0] not in lst_file_repl

    shutil.copy(testdata / 'nonmem' / 'pheno_real.mod', tmp_path / 'pheno_real.mod')
    shutil.copy(testdata / 'nonmem' / 'pheno_real.ext', tmp_path / 'pheno_real.ext')

    with chdir(tmp_path):
        with open('pheno_real.lst', 'a') as f:
            f.write(lst_file_repl)

        res = read_modelfit_results('pheno_real.mod')
        runtime = res.runtime_total
        assert runtime == runtime_ref


def test_estimation_runtime_steps_pheno(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)

    assert res.estimation_runtime_iterations.iloc[0] == 0.32
    assert res.runtime_total == 4


def test_estimation_runtime_steps_pheno_mult_est(testdata, load_model_for_test):
    path = (
        testdata
        / 'nonmem'
        / 'modelfit_results'
        / 'onePROB'
        / 'multEST'
        / 'noSIM'
        / 'pheno_multEST.mod'
    )

    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert res.estimation_runtime_iterations.iloc[0] == 0.33
    assert res.estimation_runtime_iterations.iloc[1] == 2.75
    assert res.runtime_total == 7
    assert res.estimation_runtime == 0.33


def test_evaluation(testdata, load_model_for_test):
    path = (
        testdata
        / 'nonmem'
        / 'modelfit_results'
        / 'onePROB'
        / 'multEST'
        / 'noSIM'
        / 'pheno_multEST.mod'
    )

    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)

    assert round(res.ofv, 3) == 729.955
    assert res.minimization_successful_iterations.iloc[-1]
    assert not res.minimization_successful


def test_serialization(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'models' / 'mox_2comp.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    res_json = res.to_json()
    res_decode = read_results(res_json)

    assert res.parameter_estimates.equals(res_decode.parameter_estimates)
    assert res.log.to_dataframe().equals(res_decode.log.to_dataframe())


def test_empty_results(testdata, pheno):
    res = parse_modelfit_results(
        pheno, testdata / 'nonmem' / 'errors' / 'no_header_error_only_iter.ext'
    )
    assert np.isnan(res.ofv)
    assert res.parameter_estimates.isnull().all()


def test_saem(testdata, load_model_for_test):
    path = testdata / 'nonmem' / 'modelfit_results' / 'saem' / 'pheno_saem.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert res.individual_ofv is not None
    assert res.ofv == 366.33573391569922
    assert res.individual_eta_samples.loc[(1, 1), 'ETA_1'] == -0.113378


def test_derivative_results(testdata, load_model_for_test):
    path = testdata / "nonmem" / "linearize" / "linearize_dir1" / "scm_dir1" / "derivatives.mod"
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert model.execution_steps[0].derivatives != ()
    assert res.derivatives is not None

    derivatives = model.execution_steps[0].derivatives
    derivative_names = tuple(tuple(map(str, d)) for d in derivatives)
    derivative_names = tuple(";".join(d) for d in derivative_names)
    assert all(d in res.derivatives.columns for d in derivative_names)
