from pharmpy import Model


def test_modelfit(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert model
