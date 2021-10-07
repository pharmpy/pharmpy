import pytest

from pharmpy.estimation import EstimationMethod


def test_estimation_method():
    a = EstimationMethod('foce', cov=True)
    assert a.method == 'FOCE'
    assert a.cov

    with pytest.raises(ValueError):
        EstimationMethod('sklarfs')

    a.method = 'fo'
    assert a.method == 'FO'

    assert a == EstimationMethod('fo', interaction=False, cov=True)
    assert repr(a) == 'EstimationMethod("FO", interaction=False, cov=True, options=None)'


def test_append_options():
    a = EstimationMethod('foce')

    a.append_options({'EONLY': 1})
    assert len(a.options) == 1

    a.append_options({'PRINT': 1})
    assert len(a.options) == 2
