import pytest

from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults


def test_ofv(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    assert res.ofv == 586.27605628188053


def test_covariance(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    cov = res.covariance_matrix
    assert len(cov) == 6


def test_individual_OFV(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    iofv = res.individual_OFV
    assert len(iofv) == 59
    assert pytest.approx(iofv[1], 1e-15) == 5.9473520242962552
    assert pytest.approx(iofv[57], 1e-15) == 5.6639479151436394
    assert res.plot_iofv_vs_iofv(res)


def test_parameter_estimates(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    pe = res.parameter_estimates
    print(pe)
    assert len(pe) == 6
    assert pe['THETA(1)'] == 4.69555e-3
    assert pe['OMEGA(2,2)'] == 2.7906e-2
