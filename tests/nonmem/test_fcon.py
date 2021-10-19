from pharmpy import Model


def test_dataset(testdata):
    model = Model(testdata / 'nonmem' / 'fcon' / 'FCON')
    assert model.code.startswith('FILE')

    df = model.dataset

    nmtran = Model(testdata / 'nonmem' / 'pheno_real.mod')

    assert list(df['WGT']) == list(nmtran.dataset['WGT'])
