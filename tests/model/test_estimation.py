import pytest

from pharmpy.basic import Expr
from pharmpy.internals.immutable import frozenmapping
from pharmpy.model import EstimationStep, ExecutionSteps, SimulationStep


def test_init():
    a = EstimationStep.create('foce', tool_options={'opt1': 23})
    assert a.tool_options['opt1'] == 23
    a = EstimationStep.create('foce', solver='lsoda')
    assert a.solver == 'LSODA'
    with pytest.raises(ValueError):
        EstimationStep.create('foce', solver='unknownsolverz')
    steps = ExecutionSteps.create()
    assert len(steps) == 0


def test_estimation_method():
    a = EstimationStep.create('foce', parameter_uncertainty_method='sandwich')
    assert a.method == 'FOCE'
    assert a.parameter_uncertainty_method == 'SANDWICH'

    with pytest.raises(ValueError):
        EstimationStep.create('sklarfs')

    a = a.replace(method='fo')
    assert a.method == 'FO'
    assert a.parameter_uncertainty_method == 'SANDWICH'

    assert a == EstimationStep.create(
        'fo', interaction=False, parameter_uncertainty_method='sandwich'
    )
    assert (
        repr(a)
        == "EstimationStep('FO', interaction=False, parameter_uncertainty_method='SANDWICH', evaluation=False, "
        "maximum_evaluations=None, laplace=False, isample=None, niter=None, auto=None, "
        "keep_every_nth_iter=None, individual_eta_samples=False, solver=None, solver_rtol=None, solver_atol=None, "
        "tool_options={})"
    )

    with pytest.raises(ValueError):
        EstimationStep.create('fo', maximum_evaluations=0)

    with pytest.raises(ValueError):
        EstimationStep.create('fo', parameter_uncertainty_method='x')


def test_repr():
    steps = ExecutionSteps()
    assert steps._repr_html_() == 'ExecutionSteps()'
    assert repr(steps) == 'ExecutionSteps()'

    a = EstimationStep.create('foce')
    steps = ExecutionSteps.create([a])
    assert type(steps._repr_html_()) == str
    assert type(repr(steps)) == str


def test_eq():
    s1 = ExecutionSteps()
    s2 = ExecutionSteps()
    assert s1 == s2
    a = EstimationStep.create('foce')
    s3 = ExecutionSteps.create([a, a])
    assert s1 != s3
    b = EstimationStep.create('fo')
    s4 = ExecutionSteps.create([a, b])
    assert s3 != s4
    assert s3 != 'x'


def test_len():
    s1 = ExecutionSteps()
    assert len(s1) == 0
    a = EstimationStep.create('foce')
    s2 = ExecutionSteps.create([a, a])
    assert len(s2) == 2


def test_radd():
    a = EstimationStep.create('foce')
    s2 = ExecutionSteps.create([a, a])
    b = EstimationStep.create('fo')
    conc = b + s2
    assert len(conc) == 3
    conc = (b,) + s2
    assert len(conc) == 3


def test_add():
    a = EstimationStep.create('foce')
    s2 = ExecutionSteps.create([a, a])
    b = EstimationStep.create('fo')
    s3 = ExecutionSteps.create([a, b])
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
    s1 = ExecutionSteps.create([a, b])
    s2 = ExecutionSteps.create([a])
    assert hash(s1) != hash(s2)


def test_dict():
    a = EstimationStep.create('foce')
    d = a.to_dict()
    assert d == {
        'method': 'FOCE',
        'interaction': False,
        'parameter_uncertainty_method': None,
        'evaluation': False,
        'maximum_evaluations': None,
        'laplace': False,
        'isample': None,
        'niter': None,
        'auto': None,
        'class': 'EstimationStep',
        'keep_every_nth_iter': None,
        'solver': None,
        'solver_rtol': None,
        'solver_atol': None,
        'tool_options': {},
        'derivatives': (),
        'predictions': (),
        'residuals': (),
        'individual_eta_samples': False,
    }
    step2 = EstimationStep.from_dict(d)
    assert step2 == a

    b = EstimationStep.create('fo')
    s1 = ExecutionSteps.create([a, b])
    d = s1.to_dict()
    assert d == {
        'steps': (
            {
                'method': 'FOCE',
                'interaction': False,
                'parameter_uncertainty_method': None,
                'evaluation': False,
                'maximum_evaluations': None,
                'laplace': False,
                'isample': None,
                'niter': None,
                'auto': None,
                'class': 'EstimationStep',
                'keep_every_nth_iter': None,
                'solver': None,
                'solver_rtol': None,
                'solver_atol': None,
                'tool_options': {},
                'derivatives': (),
                'predictions': (),
                'residuals': (),
                'individual_eta_samples': False,
            },
            {
                'method': 'FO',
                'interaction': False,
                'parameter_uncertainty_method': None,
                'evaluation': False,
                'maximum_evaluations': None,
                'laplace': False,
                'isample': None,
                'niter': None,
                'auto': None,
                'class': 'EstimationStep',
                'keep_every_nth_iter': None,
                'solver': None,
                'solver_rtol': None,
                'solver_atol': None,
                'tool_options': {},
                'derivatives': (),
                'predictions': (),
                'residuals': (),
                'individual_eta_samples': False,
            },
        )
    }
    s2 = ExecutionSteps.from_dict(d)
    assert s1 == s2

    ss = SimulationStep(n=23)
    s3 = s1 + ss
    d = s3.to_dict()
    s4 = ExecutionSteps.from_dict(d)
    assert s3 == s4


def test_getitem():
    a = EstimationStep.create('foce')
    b = EstimationStep.create('fo')
    s = ExecutionSteps.create([a, b])
    assert s[0].method == 'FOCE'
    assert s[1].method == 'FO'

    assert len(s[1:]) == 1


def test_properties():
    d = tuple(map(Expr.symbol, ('EPS_1',)))
    a = EstimationStep.create('foce', derivatives=(d,))
    assert a.derivatives == (d,)

    d = tuple(map(Expr.symbol, ('ETA_1',)))
    a = EstimationStep.create('foce', derivatives=(d,))
    assert a.derivatives == (d,)

    d = tuple(map(Expr.symbol, ('EPS_1', 'ETA_1')))
    a = EstimationStep.create('foce', derivatives=(d,))
    assert a.derivatives == (d,)

    d = tuple(tuple(map(Expr.symbol, params)) for params in (('EPS_1', 'ETA_1'), ('ETA_1',)))
    d2 = list(list(map(Expr.symbol, params)) for params in (('EPS_1', 'ETA_1'), ('ETA_1',)))
    a = EstimationStep.create('foce', derivatives=d)
    b = EstimationStep.create('foce', derivatives=d2)
    assert a.derivatives == b.derivatives == d

    d = tuple(tuple(map(Expr.symbol, params)) for params in (('ETA_1',), ('EPS_1', 'ETA_1')))
    a = EstimationStep.create('foce', derivatives=d)
    assert a.derivatives == EstimationStep._canonicalize_derivatives(d)

    with pytest.raises(TypeError, match="Given derivatives cannot be converted to tuple of tuples"):
        a = EstimationStep.create('foce', derivatives=13)
    with pytest.raises(TypeError, match="Given derivatives cannot be converted to tuple of tuples"):
        a = EstimationStep.create('foce', derivatives=(13,))
    with pytest.raises(TypeError, match="Each derivative argument must be a symbol of type"):
        a = EstimationStep.create('foce', derivatives=((13,),))

    a1 = EstimationStep.create('foce', predictions=('PRED',))
    a2 = EstimationStep.create('foce', predictions=['PRED'])
    assert a1.predictions == a2.predictions == ('PRED',)
    a1 = EstimationStep.create('foce', residuals=('CWRES',))
    a2 = EstimationStep.create('foce', residuals=['CWRES'])
    assert a1.residuals == a2.residuals == ('CWRES',)

    with pytest.raises(TypeError, match="Predictions could not be converted to tuple"):
        a = EstimationStep.create('foce', predictions=13)
    with pytest.raises(TypeError, match="Residuals could not be converted to tuple"):
        a = EstimationStep.create('foce', residuals=13)

    e1 = EstimationStep.create('foce', individual_eta_samples=True)
    assert e1.individual_eta_samples
    e2 = EstimationStep.create('foce')
    assert not e2.individual_eta_samples


def test_replace():
    a = EstimationStep.create('foce')
    b = a.replace(method='fo')
    assert b.method == 'FO'
    c = a.replace(solver_atol=0.01)
    assert c.solver_atol == 0.01

    steps1 = ExecutionSteps((a,))
    steps2 = ExecutionSteps((b,))
    steps3 = steps1.replace(steps=[steps2])
    assert len(steps3) == 1


def test_simulation_step():
    ss = SimulationStep(n=23)
    assert ss.n == 23

    with pytest.raises(ValueError):
        SimulationStep.create(n=0)

    ss = SimulationStep.create(n=2)
    assert ss.n == 2
    ss = ss.replace(n=19)
    assert ss.n == 19

    ss = SimulationStep(seed=64206)
    assert ss.seed == 64206

    ss1 = SimulationStep()
    ss2 = SimulationStep(n=2)
    assert ss1 != ss2
    assert ss1 == ss1
    assert hash(ss1) != hash(ss2)

    d = ss2.to_dict()
    assert d == {
        'class': 'SimulationStep',
        'n': 2,
        'seed': 1234,
        'solver': None,
        'solver_atol': None,
        'solver_rtol': None,
        'tool_options': {},
    }
    assert ss2 == SimulationStep.from_dict(d)
    d['tool_options'] = frozenmapping({})
    assert ss2 == SimulationStep.from_dict(d)

    assert (
        repr(ss2)
        == 'SimulationStep(n=2, seed=1234, solver=None, solver_rtol=None, solver_atol=None, tool_options={})'
    )
