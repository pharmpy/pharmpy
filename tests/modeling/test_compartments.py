from pharmpy.modeling import add_bioavailability, add_lag_time, get_bioavailability, get_lag_times


def test_get_lag_times(pheno, load_model_for_test, testdata):
    lags = get_lag_times(pheno)
    assert lags == {}

    pheno_lag = add_lag_time(pheno)
    lags = get_lag_times(pheno_lag)
    assert len(lags) == 1

    model_pred = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    lags = get_lag_times(model_pred)
    assert lags == {}


def test_get_bioavailability(pheno, load_model_for_test, testdata):
    fs = get_bioavailability(pheno)
    assert fs == {}

    pheno_bioavail = add_bioavailability(pheno)
    fs = get_bioavailability(pheno_bioavail)
    assert len(fs) == 1

    model_pred = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    lags = get_bioavailability(model_pred)
    assert lags == {}
