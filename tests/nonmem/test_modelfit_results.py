import pytest

from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults


def test_ofv(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    assert res.ofv == 586.27605628188053


def test_tool_files(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    names = [str(p.name) for p in res.tool_files]
    assert names == ['pheno_real.lst', 'pheno_real.ext', 'pheno_real.cov', 'pheno_real.cor', 'pheno_real.coi', 'pheno_real.phi']


def test_covariance(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    cov = res.covariance_matrix
    assert len(cov) == 6
    assert pytest.approx(cov.loc['THETA(1)', 'THETA(1)'], 1e-13) == 4.41151E-08
    assert pytest.approx(cov.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 7.17184E-05


def test_information(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    m = res.information_matrix
    assert len(m) == 6
    assert pytest.approx(m.loc['THETA(1)', 'THETA(1)'], 1e-13) == 2.99556E+07
    assert pytest.approx(m.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == -2.80082E+03


def test_correlation(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    corr = res.correlation_matrix
    assert len(corr) == 6
    assert corr.loc['THETA(1)', 'THETA(1)'] == 1.0
    assert pytest.approx(corr.loc['OMEGA(2,2)', 'THETA(2)'], 1e-13) == 3.56662E-01


def test_standard_errors(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    ses = res.standard_errors
    assert len(ses) == 6
    assert pytest.approx(ses['THETA(1)'], 1e-13) == 2.10036E-04


def test_individual_ofv(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    iofv = res.individual_ofv
    assert len(iofv) == 59
    assert pytest.approx(iofv[1], 1e-15) == 5.9473520242962552
    assert pytest.approx(iofv[57], 1e-15) == 5.6639479151436394
    assert res.plot_iofv_vs_iofv(res)


def test_parameter_estimates(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    pe = res.parameter_estimates
    assert len(pe) == 6
    assert pe['THETA(1)'] == 4.69555e-3
    assert pe['OMEGA(2,2)'] == 2.7906e-2
