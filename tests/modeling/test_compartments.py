from pharmpy.modeling import get_lag_times


def test_get_lag_times(pheno):
    lags = get_lag_times(pheno)
    assert lags == dict()
