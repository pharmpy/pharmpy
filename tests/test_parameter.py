import pytest
import sympy

from pharmpy.parameter import Parameter, ParameterSet
from pharmpy.symbols import symbol


@pytest.mark.parametrize(
    'name,init,lower,upper,fix',
    [
        ('THETA(1)', 23, None, None, None),
        ('X', 12, None, None, None),
        ('_NAME', 0, None, None, None),
        ('OMEGA(2,1)', 0.1, 0, None, None),
        ('TCVL', 0.23, -2, 2, None),
    ],
)
def test_initialization(name, init, lower, upper, fix):
    param = Parameter(name, init, lower, upper, fix)
    assert param.name == name
    assert param.symbol == symbol(name)
    assert param.init == init
    if lower is not None:
        assert param.lower == lower
    else:
        assert param.lower == -sympy.oo
    if upper is not None:
        assert param.upper == upper
    else:
        assert param.upper == sympy.oo
    assert param.fix == bool(fix)


@pytest.mark.parametrize(
    'name,init,lower,upper,fix',
    [
        ('OMEGA(2,1)', 0.1, 2, None, None),
        ('X', 1, 0, -1, None),
    ],
)
def test_illegal_initialization(name, init, lower, upper, fix):
    with pytest.raises(ValueError):
        Parameter(name, init, lower, upper, fix)


def test_any_boundary_near_value():
    param = Parameter('X', 2, lower=0, upper=23.8)
    assert param.any_boundary_near_value(0.007, zero_limit=0.01, significant_digits=2) is True
    assert param.any_boundary_near_value(0.007, zero_limit=0.001, significant_digits=2) is False
    assert param.any_boundary_near_value(23.2, zero_limit=0.001, significant_digits=2) is False
    assert param.any_boundary_near_value(23.2, zero_limit=0.001, significant_digits=1) is True
    assert param.any_boundary_near_value(23.5, zero_limit=0.001, significant_digits=2) is True


def test_repr():
    param = Parameter('X', 2, lower=0, upper=23)
    assert repr(param) == 'Parameter("X", 2, lower=0, upper=23, fix=False)'


def test_unconstrain():
    param = Parameter('X', 2, lower=0, upper=23)
    param.unconstrain()
    assert param.lower == -sympy.oo
    assert param.upper == sympy.oo

    fixed_param = Parameter('Y', 0, fix=True)
    fixed_param.unconstrain()
    assert fixed_param.lower == -sympy.oo
    assert fixed_param.upper == sympy.oo


def test_fix():
    param = Parameter('X', 2, lower=0, upper=23)
    assert param == Parameter('X', 2, lower=0, upper=23, fix=False)
    param.fix = True
    assert param == Parameter('X', 2, lower=0, upper=23, fix=True)


def test_init():
    param = Parameter('X', 2, lower=0, upper=23)
    with pytest.raises(ValueError):
        param.init = -1
    param.init = 22


def test_pset_index():
    p = Parameter('Y', 9)
    pset = ParameterSet((p,))
    assert len(pset) == 1
    assert pset['Y'] is p

    p2 = Parameter('Z', 5)
    pset.add(p2)

    assert len(pset) == 2

    # Check that the parameter set keeps the insertion order upon iteration
    for i, param in enumerate(pset):
        if i == 0:
            assert param is p
        else:
            assert param is p2


def test_pset_remove_fixed():
    p1 = Parameter('Y', 9, fix=False)
    p2 = Parameter('X', 3, fix=True)
    p3 = Parameter('Z', 1, fix=False)
    pset = ParameterSet([p1, p2, p3])
    pset.remove_fixed()
    assert len(pset) == 2
    assert pset['Y'] == Parameter('Y', 9)


def test_pset_names():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset = ParameterSet([p1, p2, p3])
    assert pset.names == ['Y', 'X', 'Z']
    assert pset.symbols == [symbol('Y'), symbol('X'), symbol('Z')]


def test_pset_lower_upper():
    p1 = Parameter('X', 0, lower=-1, upper=1)
    p2 = Parameter('Y', 1, lower=0)
    pset = ParameterSet([p1, p2])
    assert pset.lower == {'X': -1, 'Y': 0}
    assert pset.upper == {'X': 1, 'Y': sympy.oo}


def test_pset_inits():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset = ParameterSet([p1, p2, p3])
    pset.inits = {'X': 28}
    assert len(pset) == 3
    assert pset['X'] == Parameter('X', 28)
    assert pset['Y'] == Parameter('Y', 9)
    assert pset['Z'] == Parameter('Z', 1)

    with pytest.raises(KeyError):
        pset.inits = {'CL': 0}

    pset.inits = {'X': 0, 'Y': 2, 'Z': 5}
    assert len(pset) == 3
    assert pset['X'] == Parameter('X', 0)
    assert pset['Y'] == Parameter('Y', 2)
    assert pset['Z'] == Parameter('Z', 5)


def test_pset_nonfixed_inits():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset = ParameterSet([p1, p2, p3])
    assert pset.nonfixed_inits == {'Y': 9, 'X': 3, 'Z': 1}
    pset['X'].fix = True
    assert pset.nonfixed_inits == {'Y': 9, 'Z': 1}


def test_pset_fix():
    p1 = Parameter('Y', 9, fix=False)
    p2 = Parameter('X', 3, fix=True)
    p3 = Parameter('Z', 1, fix=False)
    pset = ParameterSet([p1, p2, p3])
    assert pset.fix == {'Y': False, 'X': True, 'Z': False}
    fixedness = {'Y': True, 'X': True, 'Z': True}
    pset.fix = fixedness
    assert pset.fix == {'Y': True, 'X': True, 'Z': True}


def test_pset_repr():
    p1 = Parameter('Y', 9, fix=False)
    pset = ParameterSet([p1])
    assert type(repr(pset)) == str
    assert type(pset._repr_html_()) == str
    pset = ParameterSet()
    assert type(repr(pset)) == str
    assert type(pset._repr_html_()) == str


def test_parameter_space():
    p1 = Parameter('Y', 9, fix=True)
    assert p1.parameter_space == sympy.FiniteSet(9)
    p2 = Parameter('X', 10, lower=0, upper=15)
    assert p2.parameter_space == sympy.Interval(0, 15)


def test_pset_eq():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset1 = ParameterSet([p1, p2, p3])
    pset2 = ParameterSet([p1, p2])
    assert pset1 != pset2
    pset3 = ParameterSet([p1, p3, p2])
    assert pset1 != pset3
    assert pset1 == pset1
