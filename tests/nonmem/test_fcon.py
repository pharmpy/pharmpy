def test_dataset(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'fcon' / 'FCON')
    assert model.code.startswith('FILE')

    df = model.dataset

    nmtran = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    assert list(df['WGT']) == list(nmtran.dataset['WGT'])
