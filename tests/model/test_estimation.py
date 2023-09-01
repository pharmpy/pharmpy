import pytest

from pharmpy.model import EstimationStep, EstimationSteps


def test_init():
    a = EstimationStep.create('foce', tool_options={'opt1': 23})
    assert a.tool_options['opt1'] == 23
    a = EstimationStep.create('foce', solver='lsoda')
    assert a.solver == 'LSODA'
    with pytest.raises(ValueError):
        EstimationStep.create('foce', solver='unknownsolverz')
    steps = EstimationSteps.create()
    assert len(steps) == 0


def test_estimation_method():
    a = EstimationStep.create('foce', uncert_method='sandwich')
    assert a.est_method == 'FOCE'
    assert a.uncert_method == 'SANDWICH'

    with pytest.raises(ValueError):
        EstimationStep.create('sklarfs')

    a = a.replace(est_method='fo')
    assert a.est_method == 'FO'
    assert a.uncert_method == 'SANDWICH'

    assert a == EstimationStep.create('fo', interaction=False, uncert_method='sandwich')
    assert (
        repr(a)
        == "EstimationStep('FO', interaction=False, uncert_method='SANDWICH', evaluation=False, "
        "maximum_evaluations=None, laplace=False, isample=None, niter=None, auto=None, "
        "keep_every_nth_iter=None, solver=None, solver_rtol=None, solver_atol=None, "
        "tool_options={})"
    )

    with pytest.raises(ValueError):
        EstimationStep.create('fo', maximum_evaluations=0)


def test_repr():
    steps = EstimationSteps()
    assert steps._repr_html_() == 'EstimationSteps()'
    assert repr(steps) == 'EstimationSteps()'

    a = EstimationStep.create('foce')
    steps = EstimationSteps.create([a])
    assert type(steps._repr_html_()) == str
    assert type(repr(steps)) == str


def test_eq():
    s1 = EstimationSteps()
    s2 = EstimationSteps()
    assert s1 == s2
    a = EstimationStep.create('foce')
    s3 = EstimationSteps.create([a, a])
    assert s1 != s3
    b = EstimationStep.create('fo')
    s4 = EstimationSteps.create([a, b])
    assert s3 != s4


def test_len():
    s1 = EstimationSteps()
    assert len(s1) == 0
    a = EstimationStep.create('foce')
    s2 = EstimationSteps.create([a, a])
    assert len(s2) == 2


def test_radd():
    a = EstimationStep.create('foce')
    s2 = EstimationSteps.create([a, a])
    b = EstimationStep.create('fo')
    conc = b + s2
    assert len(conc) == 3
    conc = (b,) + s2
    assert len(conc) == 3


def test_add():
    a = EstimationStep.create('foce')
    s2 = EstimationSteps.create([a, a])
    b = EstimationStep.create('fo')
    s3 = EstimationSteps.create([a, b])
    conc = s2 + b
    assert len(conc) == 3
    conc = s2 + (b,)
    assert len(conc) == 3
    conc = s2 + s3
    assert len(conc) == 4


def test_hash():
    a = EstimationStep.create('foce')
    b = EstimationStep.create('fo')
    assert hash(a) != hash(b)
    s1 = EstimationSteps.create([a, b])
    s2 = EstimationSteps.create([a])
    assert hash(s1) != hash(s2)


def test_dict():
    a = EstimationStep.create('foce')
    d = a.to_dict()
    assert d == {
        'est_method': 'FOCE',
        'interaction': False,
        'uncert_method': None,
        'evaluation': False,
        'maximum_evaluations': None,
        'laplace': False,
        'isample': None,
        'niter': None,
        'auto': None,
        'keep_every_nth_iter': None,
        'solver': None,
        'solver_rtol': None,
        'solver_atol': None,
        'tool_options': {},
    }
    step2 = EstimationStep.from_dict(d)
    assert step2 == a

    b = EstimationStep.create('fo')
    s1 = EstimationSteps.create([a, b])
    d = s1.to_dict()
    assert d == {
        'steps': (
            {
                'est_method': 'FOCE',
                'interaction': False,
                'uncert_method': None,
                'evaluation': False,
                'maximum_evaluations': None,
                'laplace': False,
                'isample': None,
                'niter': None,
                'auto': None,
                'keep_every_nth_iter': None,
                'solver': None,
                'solver_rtol': None,
                'solver_atol': None,
                'tool_options': {},
            },
            {
                'est_method': 'FO',
                'interaction': False,
                'uncert_method': None,
                'evaluation': False,
                'maximum_evaluations': None,
                'laplace': False,
                'isample': None,
                'niter': None,
                'auto': None,
                'keep_every_nth_iter': None,
                'solver': None,
                'solver_rtol': None,
                'solver_atol': None,
                'tool_options': {},
            },
        )
    }
    s2 = EstimationSteps.from_dict(d)
    assert s1 == s2


def test_getitem():
    a = EstimationStep.create('foce')
    b = EstimationStep.create('fo')
    s = EstimationSteps.create([a, b])
    assert s[0].est_method == 'FOCE'
    assert s[1].est_method == 'FO'

    assert len(s[1:]) == 1


def test_properties():
    a = EstimationStep.create('foce', epsilon_derivatives=['EPS(1)'])
    assert a.epsilon_derivatives == ('EPS(1)',)
    a = EstimationStep.create('foce', eta_derivatives=['ETA(1)'])
    assert a.eta_derivatives == ('ETA(1)',)
    a = EstimationStep.create('foce', predictions=['PRED'])
    assert a.predictions == ('PRED',)
    a = EstimationStep.create('foce', residuals=['CWRES'])
    assert a.residuals == ('CWRES',)


def test_replace():
    a = EstimationStep.create('foce')
    b = a.replace(est_method='fo')
    assert b.est_method == 'FO'
    c = a.replace(solver_atol=0.01)
    assert c.solver_atol == 0.01

    steps1 = EstimationSteps((a,))
    steps2 = EstimationSteps((b,))
    steps3 = steps1.replace(steps=[steps2])
    assert len(steps3) == 1
