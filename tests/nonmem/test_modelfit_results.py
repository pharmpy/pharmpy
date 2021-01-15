import pandas as pd
import pytest

import pharmpy.plugins.nonmem as nonmem
from pharmpy import Model
from pharmpy.config import ConfigurationContext
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults, simfit_results


def test_ofv(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst)
    assert res.ofv == 586.27605628188053


def test_aic_bic(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    assert model.modelfit_results.aic == 740.8947268137308
    assert model.modelfit_results.bic == 756.111852398327033103


def test_tool_files(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst)
    names = [str(p.name) for p in res.tool_files]
    assert names == [
        'pheno_real.lst',
        'pheno_real.ext',
        'pheno_real.cov',
        'pheno_real.cor',
        'pheno_real.coi',
        'pheno_real.phi',
    ]


def test_condition_number(testdata, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst)
    assert res.condition_number == pytest.approx(4.39152)

    maxeval3 = Model(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'noSIM' / 'maxeval3.mod'
    )

    assert maxeval3.modelfit_results.condition_number == 4.77532e06

    maxeval0 = Model(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'noSIM' / 'maxeval0.mod'
    )
    assert maxeval0.modelfit_results.condition_number is None


def test_sumo(testdata):
    onePROB = testdata / 'nonmem' / 'modelfit_results' / 'onePROB'
    pheno = Model(onePROB / 'oneEST' / 'noSIM' / 'pheno.mod')
    d = pheno.modelfit_results.sumo(to_string=False)
    assert 'Messages' in d.keys()
    assert 'Parameter summary' in d.keys()


def test_special_models(testdata):
    onePROB = testdata / 'nonmem' / 'modelfit_results' / 'onePROB'
    withBayes = Model(onePROB / 'multEST' / 'noSIM' / 'withBayes.mod')
    assert (
        pytest.approx(withBayes.modelfit_results.standard_errors['THETA(1)'], 1e-13) == 2.51942e00
    )
    assert (
        pytest.approx(withBayes.modelfit_results[0].standard_errors['THETA(1)'], 1e-13)
        == 3.76048e-01
    )
    assert withBayes.modelfit_results[0].minimization_successful is False
    assert withBayes.modelfit_results[1].minimization_successful is False
    assert withBayes.modelfit_results[0].covariance_step == {
        'requested': True,
        'completed': True,
        'warnings': False,
    }
    assert withBayes.modelfit_results.covariance_step == {
        'requested': True,
        'completed': True,
        'warnings': False,
    }

    maxeval0 = Model(onePROB / 'oneEST' / 'noSIM' / 'maxeval0.mod')
    assert maxeval0.modelfit_results.minimization_successful is None

    maxeval3 = Model(onePROB / 'oneEST' / 'noSIM' / 'maxeval3.mod')
    assert maxeval3.modelfit_results.minimization_successful is False
    assert maxeval3.modelfit_results.covariance_step == {
        'requested': True,
        'completed': True,
        'warnings': True,
    }

    nearbound = Model(onePROB / 'oneEST' / 'noSIM' / 'near_bounds.mod')
    correct = pd.Series(
        [False, True, False, False, False, False, False, False, True, True, False],
        index=[
            'THETA(1)',
            'THETA(2)',
            'THETA(3)',
            'THETA(4)',
            'OMEGA(1,1)',
            'OMEGA(2,1)',
            'OMEGA(2,2)',
            'OMEGA(3,3)',
            'OMEGA(4,4)',
            'OMEGA(6,6)',
            'SIGMA(1,1)',
        ],
    )
    pd.testing.assert_series_equal(nearbound.modelfit_results.near_bounds(), correct)


def test_covariance(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        cov = res.covariance_matrix
        assert len(cov) == 6
        assert pytest.approx(cov.loc['THETA(1)', 'THETA(1)'], 1e-13) == 4.41151e-08
        assert pytest.approx(cov.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 7.17184e-05
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        cov = res.covariance_matrix
        assert len(cov) == 6
        assert pytest.approx(cov.loc['PTVCL', 'PTVCL'], 1e-13) == 4.41151e-08
        assert pytest.approx(cov.loc['IVV', 'PTVV'], 1e-13) == 7.17184e-05


def test_information(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        m = res.information_matrix
        assert len(m) == 6
        assert pytest.approx(m.loc['THETA(1)', 'THETA(1)'], 1e-13) == 2.99556e07
        assert pytest.approx(m.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == -2.80082e03
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        m = res.information_matrix
        assert len(m) == 6
        assert pytest.approx(m.loc['PTVCL', 'PTVCL'], 1e-13) == 2.99556e07
        assert pytest.approx(m.loc['IVV', 'PTVV'], 1e-13) == -2.80082e03


def test_correlation(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        corr = res.correlation_matrix
        assert len(corr) == 6
        assert corr.loc['THETA(1)', 'THETA(1)'] == 1.0
        assert pytest.approx(corr.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 3.56662e-01
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        corr = res.correlation_matrix
        assert len(corr) == 6
        assert corr.loc['PTVCL', 'PTVV'] == 0.00709865
        assert pytest.approx(corr.loc['IVV', 'PTVV'], 1e-13) == 3.56662e-01


def test_standard_errors(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        ses = res.standard_errors
        assert len(ses) == 6
        assert pytest.approx(ses['THETA(1)'], 1e-13) == 2.10036e-04
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        ses = res.standard_errors
        assert len(ses) == 6
        assert pytest.approx(ses['PTVCL'], 1e-13) == 2.10036e-04


def test_individual_ofv(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, model=pheno)
    iofv = res.individual_ofv
    assert len(iofv) == 59
    assert pytest.approx(iofv[1], 1e-15) == 5.9473520242962552
    assert pytest.approx(iofv[57], 1e-15) == 5.6639479151436394
    assert res.plot_iofv_vs_iofv(res)


def test_individual_estimates(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, model=pheno)
    ie = res.individual_estimates
    assert len(ie) == 59
    assert pytest.approx(ie['ETA(1)'][1], 1e-15) == -0.0438608
    assert pytest.approx(ie['ETA(2)'][1], 1e-15) == 0.00543031
    assert pytest.approx(ie['ETA(1)'][28], 1e-15) == 7.75957e-04
    assert pytest.approx(ie['ETA(2)'][28], 1e-15) == 8.32311e-02


def test_individual_shrinkage(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, model=pheno)
    ishr = res.individual_shrinkage
    assert len(ishr) == 59
    assert pytest.approx(ishr['ETA(1)'][1], 1e-15) == 0.84778949807160287


def test_eta_shrinkage(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, model=pheno)
    shrinkage = res.eta_shrinkage()
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA(1)'], 0.0001) == 7.2048e01 / 100
    assert pytest.approx(shrinkage['ETA(2)'], 0.0001) == 2.4030e01 / 100
    shrinkage = res.eta_shrinkage(sd=True)
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA(1)'], 0.0001) == 4.7130e01 / 100
    assert pytest.approx(shrinkage['ETA(2)'], 0.0001) == 1.2839e01 / 100


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
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        pe = res.parameter_estimates
        assert len(pe) == 6
        assert pe['THETA(1)'] == 4.69555e-3
        assert pe['OMEGA(2,2)'] == 2.7906e-2
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        pe = res.parameter_estimates
        assert len(pe) == 6
        assert pe['PTVCL'] == 4.69555e-3
        assert pe['IVV'] == 2.7906e-2


def test_simfit(testdata):
    model = Model(testdata / 'nonmem' / 'modelfit_results' / 'simfit' / 'sim-1.mod')
    results = simfit_results(model)
    assert len(results) == 3
    assert results[1].ofv == 565.8490436434297
    assert results[2].ofv == 570.7344011414535


def test_residuals(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    df = model.modelfit_results.residuals
    assert len(df) == 155
    assert list(df.columns) == ['RES', 'CWRES']
    assert df['RES'][1.0, 2.0] == -0.67071
    assert df['CWRES'][1.0, 2.0] == -0.401100
