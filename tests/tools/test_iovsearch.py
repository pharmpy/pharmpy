from pharmpy import Model
from pharmpy.modeling import add_iov, remove_iov
from pharmpy.symbols import symbol as S
from pharmpy.tools.iovsearch.tool import _get_iiv_etas_with_corresponding_iov


def test_iovsearch_github_issues_976(testdata):

    m = Model.create_model(testdata / 'nonmem' / 'pheno_multivariate_piecewise.mod')
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
