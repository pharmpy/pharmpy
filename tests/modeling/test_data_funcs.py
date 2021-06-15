from pharmpy import Model
from pharmpy.modeling import ninds, nobs, nobsi


def test_ninds(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert ninds(model) == 59


def test_nobs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert nobs(model) == 155
    assert list(nobsi(model)) == [
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        3,
        1,
        3,
        2,
        4,
        2,
        3,
        3,
        4,
        3,
        3,
        3,
        2,
        3,
        3,
        6,
        2,
        2,
        1,
        1,
        2,
        1,
        3,
        2,
        2,
        2,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        1,
        3,
        3,
        1,
        1,
        5,
        3,
        4,
        3,
        3,
        2,
        4,
        1,
        1,
        2,
        3,
        3,
    ]
