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
    assert (
        repr(a) == 'EstimationMethod("FO", interaction=False, cov=True, evaluation=False, '
        'maximum_evaluations=None, laplace=False, tool_options=None)'
    )

    with pytest.raises(ValueError):
        EstimationMethod('fo', maximum_evaluations=0)


def test_append_options():
    a = EstimationMethod('foce')

    a.append_tool_options({'EONLY': 1})
    assert len(a.tool_options) == 1

    a.append_tool_options({'PRINT': 1})
    assert len(a.tool_options) == 2


def test_copy(datadir):
    a = EstimationMethod('foce', cov=True)
    b = a.copy()
    assert id(a) != id(b)
    assert b.method == 'FOCE'
