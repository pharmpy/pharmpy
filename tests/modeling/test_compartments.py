from pharmpy.modeling import get_bioavailability, get_lag_times


def test_get_lag_times(pheno):
    lags = get_lag_times(pheno)
    assert lags == {}


def test_get_bioavailability(pheno):
    fs = get_bioavailability(pheno)
    assert fs == {}
