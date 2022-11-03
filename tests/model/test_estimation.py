import pytest

from pharmpy.model import EstimationStep, EstimationSteps


def test_init():
    a = EstimationStep('foce', tool_options={'opt1': 23})
    assert a.tool_options['opt1'] == 23
    a = EstimationStep('foce', solver='lsoda')
    assert a.solver == 'LSODA'
    with pytest.raises(ValueError):
        EstimationStep('foce', solver='unknownsolverz')


def test_estimation_method():
    a = EstimationStep('foce', cov=True)
    assert a.method == 'FOCE'
    assert a.cov

    with pytest.raises(ValueError):
        EstimationStep('sklarfs')

    a = a.derive(method='fo')
    assert a.method == 'FO'

    assert a == EstimationStep('fo', interaction=False, cov=True)
    assert (
        repr(a) == 'EstimationStep("FO", interaction=False, cov=True, evaluation=False, '
        'maximum_evaluations=None, laplace=False, isample=None, niter=None, auto=None, '
        'keep_every_nth_iter=None, solver=None, solver_rtol=None, solver_atol=None, '
        'tool_options={})'
    )

    with pytest.raises(ValueError):
        EstimationStep('fo', maximum_evaluations=0)


def test_append_options():
    a = EstimationStep('foce')

    a.tool_options.update({'EONLY': 1})
    assert len(a.tool_options) == 1

    a.tool_options.update({'PRINT': 1})
    assert len(a.tool_options) == 2


def test_repr():
    steps = EstimationSteps()
    assert steps._repr_html_() == 'EstimationSteps()'
    assert repr(steps) == 'EstimationSteps()'

    a = EstimationStep('foce')
    steps = EstimationSteps([a])
    assert type(steps._repr_html_()) == str
    assert type(repr(steps)) == str


def test_eq():
    s1 = EstimationSteps()
    s2 = EstimationSteps()
    assert s1 == s2
    a = EstimationStep('foce')
    s3 = EstimationSteps([a, a])
    assert s1 != s3
    b = EstimationStep('fo')
    s4 = EstimationSteps([a, b])
    assert s3 != s4


def test_len():
    s1 = EstimationSteps()
    assert len(s1) == 0
    a = EstimationStep('foce')
    s2 = EstimationSteps([a, a])
    assert len(s2) == 2


def test_radd():
    a = EstimationStep('foce')
    s2 = EstimationSteps([a, a])
    b = EstimationStep('fo')
    conc = b + s2
    assert len(conc) == 3
    conc = (b,) + s2
    assert len(conc) == 3


def test_add():
    a = EstimationStep('foce')
    s2 = EstimationSteps([a, a])
    b = EstimationStep('fo')
    s3 = EstimationSteps([a, b])
    conc = s2 + b
    assert len(conc) == 3
    conc = s2 + (b,)
    assert len(conc) == 3
    conc = s2 + s3
    assert len(conc) == 4


def test_getitem():
    a = EstimationStep('foce')
    b = EstimationStep('fo')
    s = EstimationSteps([a, b])
    assert s[0].method == 'FOCE'
    assert s[1].method == 'FO'

    assert len(s[1:]) == 1


def test_properties():
    a = EstimationStep('foce', epsilon_derivatives=['EPS(1)'])
    assert a.epsilon_derivatives == ['EPS(1)']
    a = EstimationStep('foce', eta_derivatives=['ETA(1)'])
    assert a.eta_derivatives == ['ETA(1)']
    a = EstimationStep('foce', predictions=['PRED'])
    assert a.predictions == ['PRED']
    a = EstimationStep('foce', residuals=['CWRES'])
    assert a.residuals == ['CWRES']


def test_derive():
    a = EstimationStep('foce')
    b = a.derive(method='fo')
    assert b.method == 'FO'
    c = a.derive(solver_atol=0.01)
    assert c.solver_atol == 0.01
