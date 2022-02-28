import re

import pandas as pd
import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

import pharmpy.plugins.nonmem as nonmem
from pharmpy import Model
from pharmpy.config import ConfigurationContext
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults, simfit_results


def test_ofv(pheno):
    res = pheno.modelfit_results
    assert res.ofv == 586.27605628188053


def test_tool_files(pheno):
    res = pheno.modelfit_results
    names = [str(p.name) for p in res.tool_files]
    assert names == [
        'pheno_real.lst',
        'pheno_real.ext',
        'pheno_real.cov',
        'pheno_real.cor',
        'pheno_real.coi',
        'pheno_real.phi',
    ]


def test_special_models(testdata):
    onePROB = testdata / 'nonmem' / 'modelfit_results' / 'onePROB'
    withBayes = Model.create_model(onePROB / 'multEST' / 'noSIM' / 'withBayes.mod')
    assert (
        pytest.approx(withBayes.modelfit_results.standard_errors['THETA(1)'], 1e-13) == 2.51942e00
    )
    assert (
        pytest.approx(withBayes.modelfit_results[0].standard_errors['THETA(1)'], 1e-13)
        == 3.76048e-01
    )
    assert withBayes.modelfit_results[0].minimization_successful is False
    assert withBayes.modelfit_results[1].minimization_successful is False

    maxeval0 = Model.create_model(onePROB / 'oneEST' / 'noSIM' / 'maxeval0.mod')
    assert maxeval0.modelfit_results.minimization_successful is None

    maxeval3 = Model.create_model(onePROB / 'oneEST' / 'noSIM' / 'maxeval3.mod')
    assert maxeval3.modelfit_results.minimization_successful is False


def test_covariance(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names=['basic']):
        res = Model.create_model(pheno_path).modelfit_results
        cov = res.covariance_matrix
        assert len(cov) == 6
        assert pytest.approx(cov.loc['THETA(1)', 'THETA(1)'], 1e-13) == 4.41151e-08
        assert pytest.approx(cov.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 7.17184e-05
    with ConfigurationContext(nonmem.conf, parameter_names=['comment', 'basic']):
        res = Model.create_model(pheno_path).modelfit_results
        cov = res.covariance_matrix
        assert len(cov) == 6
        assert pytest.approx(cov.loc['PTVCL', 'PTVCL'], 1e-13) == 4.41151e-08
        assert pytest.approx(cov.loc['IVV', 'PTVV'], 1e-13) == 7.17184e-05


def test_information(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names=['basic']):
        res = Model.create_model(pheno_path).modelfit_results
        m = res.information_matrix
        assert len(m) == 6
        assert pytest.approx(m.loc['THETA(1)', 'THETA(1)'], 1e-13) == 2.99556e07
        assert pytest.approx(m.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == -2.80082e03
    with ConfigurationContext(nonmem.conf, parameter_names=['comment', 'basic']):
        res = Model.create_model(pheno_path).modelfit_results
        m = res.information_matrix
        assert len(m) == 6
        assert pytest.approx(m.loc['PTVCL', 'PTVCL'], 1e-13) == 2.99556e07
        assert pytest.approx(m.loc['IVV', 'PTVV'], 1e-13) == -2.80082e03


def test_correlation(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names=['basic']):
        res = Model.create_model(pheno_path).modelfit_results
        corr = res.correlation_matrix
        assert len(corr) == 6
        assert corr.loc['THETA(1)', 'THETA(1)'] == 1.0
        assert pytest.approx(corr.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 3.56662e-01
    with ConfigurationContext(nonmem.conf, parameter_names=['comment', 'basic']):
        res = Model.create_model(pheno_path).modelfit_results
        corr = res.correlation_matrix
        assert len(corr) == 6
        assert corr.loc['PTVCL', 'PTVV'] == 0.00709865
        assert pytest.approx(corr.loc['IVV', 'PTVV'], 1e-13) == 3.56662e-01


def test_standard_errors(testdata, pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names=['basic']):
        res = Model.create_model(pheno_path).modelfit_results
        ses = res.standard_errors
        assert len(ses) == 6
        assert pytest.approx(ses['THETA(1)'], 1e-13) == 2.10036e-04
        ses_sd = res.standard_errors_sdcorr
        correct = pd.Series(
            {
                'THETA(1)': 0.000210036,
                'THETA(2)': 0.0268952,
                'THETA(3)': 0.0837623,
                'OMEGA(1,1)': 0.0391526,
                'OMEGA(2,2)': 0.0223779,
                'SIGMA(1,1)': 0.00990444,
            }
        )
        correct.name = 'SE'
        pd.testing.assert_series_equal(ses_sd, correct)

    with ConfigurationContext(nonmem.conf, parameter_names=['comment', 'basic']):
        res = Model.create_model(pheno_path).modelfit_results
        ses = res.standard_errors
        assert len(ses) == 6
        assert pytest.approx(ses['PTVCL'], 1e-13) == 2.10036e-04

        ses_sd = res.standard_errors_sdcorr
        correct = pd.Series(
            {
                'PTVCL': 0.000210036,
                'PTVV': 0.0268952,
                'THETA(3)': 0.0837623,
                'IVCL': 0.0391526,
                'IVV': 0.0223779,
                'SIGMA(1,1)': 0.00990444,
            }
        )
        correct.name = 'SE'
        pd.testing.assert_series_equal(ses_sd, correct)


def test_individual_ofv(pheno):
    iofv = pheno.modelfit_results.individual_ofv
    assert len(iofv) == 59
    assert pytest.approx(iofv[1], 1e-15) == 5.9473520242962552
    assert pytest.approx(iofv[57], 1e-15) == 5.6639479151436394


def test_individual_estimates(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, model=pheno)
    ie = res.individual_estimates
    assert len(ie) == 59
    assert pytest.approx(ie['ETA(1)'][1], 1e-15) == -0.0438608
    assert pytest.approx(ie['ETA(2)'][1], 1e-15) == 0.00543031
    assert pytest.approx(ie['ETA(1)'][28], 1e-15) == 7.75957e-04
    assert pytest.approx(ie['ETA(2)'][28], 1e-15) == 8.32311e-02


def test_individual_estimates_covariance(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, model=pheno)
    cov = res.individual_estimates_covariance
    assert len(cov) == 59
    names = ['ETA(1)', 'ETA(2)']
    correct = pd.DataFrame(
        [[2.48833e-02, -2.99920e-03], [-2.99920e-03, 7.15713e-03]], index=names, columns=names
    )
    pd.testing.assert_frame_equal(cov[1], correct)
    correct2 = pd.DataFrame(
        [[2.93487e-02, -1.95747e-04], [-1.95747e-04, 8.94118e-03]], index=names, columns=names
    )
    pd.testing.assert_frame_equal(cov[43], correct2)


def test_parameter_estimates(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names=['basic']):
        res = Model.create_model(pheno_path).modelfit_results
        pe = res.parameter_estimates
        assert len(pe) == 6
        assert pe['THETA(1)'] == 4.69555e-3
        assert pe['OMEGA(2,2)'] == 2.7906e-2
        pe_sd = res.parameter_estimates_sdcorr
        correct = pd.Series(
            {
                'THETA(1)': 0.00469555,
                'THETA(2)': 0.984258,
                'THETA(3)': 0.158920,
                'OMEGA(1,1)': 0.171321,
                'OMEGA(2,2)': 0.167051,
                'SIGMA(1,1)': 0.115069,
            }
        )
        correct.name = 'estimates'
        pd.testing.assert_series_equal(pe_sd, correct)

    with ConfigurationContext(nonmem.conf, parameter_names=['comment', 'basic']):
        res = Model.create_model(pheno_path).modelfit_results
        pe = res.parameter_estimates
        assert len(pe) == 6
        assert pe['PTVCL'] == 4.69555e-3
        assert pe['IVV'] == 2.7906e-2


def test_simfit(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modelfit_results' / 'simfit' / 'sim-1.mod')
    results = simfit_results(model)
    assert len(results) == 3
    assert results[1].ofv == 565.84904364342981
    assert results[2].ofv == 570.73440114145342


def test_residuals(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real.mod')
    df = model.modelfit_results.residuals
    assert len(df) == 155
    assert list(df.columns) == ['RES', 'CWRES']
    assert df['RES'][1.0, 2.0] == -0.67071
    assert df['CWRES'][1.0, 2.0] == -0.401100


def test_predictions(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real.mod')
    df = model.modelfit_results.predictions
    assert len(df) == 744
    assert list(df.columns) == ['IPRED', 'PRED']
    assert df['PRED'][1.0, 0.0] == 18.143


def test_runtime_total(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real.mod')
    runtime = model.modelfit_results.runtime_total
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
def test_runtime_different_formats(testdata, starttime, endtime, runtime_ref):
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

    with Patcher(additional_skip_names=['pkgutil']) as patcher:
        fs = patcher.fs
        fs.add_real_file(testdata / 'nonmem' / 'pheno_real.mod', target_path='pheno_real.mod')
        fs.add_real_file(testdata / 'nonmem' / 'pheno_real.ext', target_path='pheno_real.ext')

        with open('pheno_real.lst', 'a') as f:
            f.write(lst_file_repl)

        model = Model.create_model('pheno_real.mod')
        runtime = model.modelfit_results.runtime_total
        assert runtime == runtime_ref


def test_estimation_runtime_steps(pheno_path, testdata):
    model = Model.create_model(pheno_path)

    res = model.modelfit_results
    assert res[0].estimation_runtime == 0.32
    assert res.runtime_total == 4

    model = Model.create_model(
        testdata
        / 'nonmem'
        / 'modelfit_results'
        / 'onePROB'
        / 'multEST'
        / 'noSIM'
        / 'pheno_multEST.mod'
    )
    res = model.modelfit_results
    assert res[0].estimation_runtime == 0.33
    assert res[1].estimation_runtime == 2.75
    assert res.runtime_total == 7
    assert res.estimation_runtime == 0.33


def test_evaluation(testdata):
    model = Model.create_model(
        testdata
        / 'nonmem'
        / 'modelfit_results'
        / 'onePROB'
        / 'multEST'
        / 'noSIM'
        / 'pheno_multEST.mod'
    )
    res = model.modelfit_results

    assert round(res.ofv, 3) == 729.955
    assert res[-1].minimization_successful
    assert not res.minimization_successful
