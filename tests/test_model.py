from pharmpy import Model


def test_create_symbol(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    symbol = model.create_symbol(stem='ETAT')
    assert symbol.name == 'ETAT1'
