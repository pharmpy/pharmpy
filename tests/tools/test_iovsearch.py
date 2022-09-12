from sympy import Symbol as S

from pharmpy.modeling import add_iov, remove_iov
from pharmpy.tools.iovsearch.tool import _get_iiv_etas_with_corresponding_iov


def test_iovsearch_github_issues_976(load_model_for_test, testdata):

    m = load_model_for_test(testdata / 'nonmem' / 'pheno_multivariate_piecewise.mod')
    assert not m.random_variables.iov
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set()

    add_iov(m, 'FA1', distribution='same-as-iiv')
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set(
        map(lambda rv: rv.symbol, m.random_variables.iiv)
    )

    remove_iov(m, 'ETA_IOV_1_1')
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == {S('ETA(2)')}

    remove_iov(m, 'ETA_IOV_2_1')
    assert not m.random_variables.iov
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set()
