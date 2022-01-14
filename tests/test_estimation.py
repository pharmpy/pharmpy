import pytest

from pharmpy.estimation import EstimationStep


def test_estimation_method():
    a = EstimationStep('foce', cov=True)
    assert a.method == 'FOCE'
    assert a.cov

    with pytest.raises(ValueError):
        EstimationStep('sklarfs')

    a.method = 'fo'
    assert a.method == 'FO'

    assert a == EstimationStep('fo', interaction=False, cov=True)
    assert (
        repr(a) == 'EstimationStep("FO", interaction=False, cov=True, evaluation=False, '
        'maximum_evaluations=None, laplace=False, isample=None, niter=None, auto=None, '
        'keep_every_nth_iter=None, tool_options={})'
    )

    with pytest.raises(ValueError):
        EstimationStep('fo', maximum_evaluations=0)


def test_append_options():
    a = EstimationStep('foce')

    a.tool_options.update({'EONLY': 1})
    assert len(a.tool_options) == 1

    a.tool_options.update({'PRINT': 1})
    assert len(a.tool_options) == 2


def test_copy(datadir):
    a = EstimationStep('foce', cov=True)
    b = a.copy()
    assert id(a) != id(b)
    assert b.method == 'FOCE'
