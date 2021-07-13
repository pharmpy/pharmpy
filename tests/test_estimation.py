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
    assert repr(a) == 'EstimationMethod("FO", interaction=False, cov=True, options=[])'
