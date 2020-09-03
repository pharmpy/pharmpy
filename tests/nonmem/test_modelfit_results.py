import pandas as pd
import pytest

import pharmpy.plugins.nonmem as nonmem
from pharmpy import Model
from pharmpy.config import ConfigurationContext
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults


def test_ofv(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    assert res.ofv == 586.27605628188053


def test_tool_files(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    names = [str(p.name) for p in res.tool_files]
    assert names == ['pheno_real.lst', 'pheno_real.ext', 'pheno_real.cov', 'pheno_real.cor',
                     'pheno_real.coi', 'pheno_real.phi']


def test_covariance(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        cov = res.covariance_matrix
        assert len(cov) == 6
        assert pytest.approx(cov.loc['THETA(1)', 'THETA(1)'], 1e-13) == 4.41151E-08
        assert pytest.approx(cov.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 7.17184E-05
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        cov = res.covariance_matrix
        assert len(cov) == 6
        assert pytest.approx(cov.loc['PTVCL', 'PTVCL'], 1e-13) == 4.41151E-08
        assert pytest.approx(cov.loc['IVV', 'PTVV'], 1e-13) == 7.17184E-05


def test_information(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        m = res.information_matrix
        assert len(m) == 6
        assert pytest.approx(m.loc['THETA(1)', 'THETA(1)'], 1e-13) == 2.99556E+07
        assert pytest.approx(m.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == -2.80082E+03
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        m = res.information_matrix
        assert len(m) == 6
        assert pytest.approx(m.loc['PTVCL', 'PTVCL'], 1e-13) == 2.99556E+07
        assert pytest.approx(m.loc['IVV', 'PTVV'], 1e-13) == -2.80082E+03


def test_correlation(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        corr = res.correlation_matrix
        assert len(corr) == 6
        assert corr.loc['THETA(1)', 'THETA(1)'] == 1.0
        assert pytest.approx(corr.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 3.56662E-01
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        corr = res.correlation_matrix
        assert len(corr) == 6
        assert corr.loc['PTVCL', 'PTVV'] == 0.00709865
        assert pytest.approx(corr.loc['IVV', 'PTVV'], 1e-13) == 3.56662E-01


def test_standard_errors(pheno_path):
    with ConfigurationContext(nonmem.conf, parameter_names='basic'):
        res = Model(pheno_path).modelfit_results
        ses = res.standard_errors
        assert len(ses) == 6
        assert pytest.approx(ses['THETA(1)'], 1e-13) == 2.10036E-04
    with ConfigurationContext(nonmem.conf, parameter_names='comment'):
        res = Model(pheno_path).modelfit_results
        ses = res.standard_errors
        assert len(ses) == 6
        assert pytest.approx(ses['PTVCL'], 1e-13) == 2.10036E-04


def test_individual_ofv(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1, model=pheno)
    iofv = res.individual_ofv
    assert len(iofv) == 59
    assert pytest.approx(iofv[1], 1e-15) == 5.9473520242962552
    assert pytest.approx(iofv[57], 1e-15) == 5.6639479151436394
    assert res.plot_iofv_vs_iofv(res)


def test_individual_estimates(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1, model=pheno)
    ie = res.individual_estimates
    assert len(ie) == 59
    assert pytest.approx(ie['ETA(1)'][1], 1e-15) == -0.0438608
    assert pytest.approx(ie['ETA(2)'][1], 1e-15) == 0.00543031
    assert pytest.approx(ie['ETA(1)'][28], 1e-15) == 7.75957e-04
    assert pytest.approx(ie['ETA(2)'][28], 1e-15) == 8.32311E-02


def test_individual_shrinkage(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1, model=pheno)
    ishr = res.individual_shrinkage
    assert len(ishr) == 59
    assert pytest.approx(ishr['ETA(1)'][1], 1e-15) == 0.84778949807160287


def test_eta_shrinkage(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1, model=pheno)
    shrinkage = res.eta_shrinkage()
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA(1)'], 0.0001) == 7.2048E+01 / 100
    assert pytest.approx(shrinkage['ETA(2)'], 0.0001) == 2.4030E+01 / 100
    shrinkage = res.eta_shrinkage(sd=True)
    assert len(shrinkage) == 2
    assert pytest.approx(shrinkage['ETA(1)'], 0.0001) == 4.7130E+01 / 100
    assert pytest.approx(shrinkage['ETA(2)'], 0.0001) == 1.2839E+01 / 100


def test_individual_estimates_covariance(pheno, pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1, model=pheno)
    cov = res.individual_estimates_covariance
    assert len(cov) == 59
    names = ['ETA(1)', 'ETA(2)']
    correct = pd.DataFrame([[2.48833E-02, -2.99920E-03], [-2.99920E-03, 7.15713E-03]],
                           index=names, columns=names)
    pd.testing.assert_frame_equal(cov[1], correct)
    correct2 = pd.DataFrame([[2.93487E-02, -1.95747E-04], [-1.95747E-04, 8.94118E-03]],
                            index=names, columns=names)
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
