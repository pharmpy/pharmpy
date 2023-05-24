from pharmpy.modeling import read_model


def test_dataset(load_model_for_test, testdata):
    model = read_model(testdata / 'nonmem' / 'fcon' / 'FCON')
    assert model.internals.code.startswith('FILE')

    df = model.dataset

    nmtran = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    assert list(df['WGT']) == list(nmtran.dataset['WGT'])
