import pytest

from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults


def test_individual_estimates(pheno_lst):
    res = NONMEMChainedModelfitResults(pheno_lst, 1)
    indests = res.individual_estimates
    assert len(indests) == 59
    assert pytest.approx(indests.loc[1].iOFV, 1e-15) == 5.9473520242962552
    assert pytest.approx(indests.loc[57].iOFV, 1e-15) == 5.6639479151436394
