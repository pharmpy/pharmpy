from pharmpy import Model
from pharmpy.modeling import ninds, nobs


def test_ninds(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert ninds(model) == 59


def test_nobs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert nobs(model) == 155
