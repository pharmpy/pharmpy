import pandas as pd
import pytest
import sympy
import sympy.physics.units as units

from pharmpy.parameter import Parameter, Parameters
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


def test_is_close_to_bound():
    param = Parameter('X', 2, lower=0, upper=23.8)
    assert param.is_close_to_bound(0.007, zero_limit=0.01, significant_digits=2) is True
    assert param.is_close_to_bound(0.007, zero_limit=0.001, significant_digits=2) is False
    assert param.is_close_to_bound(23.2, zero_limit=0.001, significant_digits=2) is False
    assert param.is_close_to_bound(23.2, zero_limit=0.001, significant_digits=1) is True
    assert param.is_close_to_bound(23.5, zero_limit=0.001, significant_digits=2) is True
    assert not param.is_close_to_bound()


def test_repr():
    param = Parameter('X', 2, lower=0, upper=23)
    assert repr(param) == 'Parameter("X", 2, lower=0, upper=23, fix=False)'


def test_copy():
    p1 = Parameter('X', 2, lower=0, upper=23)
    p2 = p1.copy()
    p1.init = 22
    assert p2.init == 2


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


def test_pset_init():
    p = Parameter('Y', 9)
    pset = Parameters([p])
    pset2 = Parameters(pset)
    assert len(pset2) == 1
    assert pset2['Y'].init == 9

    p2 = Parameter('Y', 12)
    with pytest.raises(ValueError):
        Parameters([p, p2])

    with pytest.raises(ValueError):
        Parameters([23])


def test_pset_getitem():
    p = Parameter('Y', 9)
    pset = Parameters((p,))
    assert len(pset) == 1
    assert pset['Y'] is p

    p2 = Parameter('Z', 5)
    pset.append(p2)

    assert len(pset) == 2

    # Check that the parameter set keeps the insertion order upon iteration
    for i, param in enumerate(pset):
        if i == 0:
            assert param is p
        else:
            assert param is p2

    assert pset[symbol('Z')] == p2

    p3 = Parameter('K', 19)
    with pytest.raises(KeyError):
        pset[p3]

    assert len(pset[[p]]) == 1


def test_pset_setitem():
    p1 = Parameter('P1', 1)
    p2 = Parameter('P2', 2)
    p3 = Parameter('P3', 3)
    ps = Parameters([p1, p2, p3])
    p4 = Parameter('P4', 4)
    ps[p1] = p4
    assert len(ps) == 3
    assert ps[0].name == 'P4'

    with pytest.raises(ValueError):
        ps[0] = 23

    p5 = Parameter('P4', 0)
    with pytest.raises(ValueError):
        ps[1] = p5


def test_pset_remove_fixed():
    p1 = Parameter('Y', 9, fix=False)
    p2 = Parameter('X', 3, fix=True)
    p3 = Parameter('Z', 1, fix=False)
    pset = Parameters([p1, p2, p3])
    pset.remove_fixed()
    assert len(pset) == 2
    assert pset['Y'] == Parameter('Y', 9)


def test_pset_names():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset = Parameters([p1, p2, p3])
    assert pset.names == ['Y', 'X', 'Z']
    assert pset.symbols == [symbol('Y'), symbol('X'), symbol('Z')]


def test_pset_lower_upper():
    p1 = Parameter('X', 0, lower=-1, upper=1)
    p2 = Parameter('Y', 1, lower=0)
    pset = Parameters([p1, p2])
    assert pset.lower == {'X': -1, 'Y': 0}
    assert pset.upper == {'X': 1, 'Y': sympy.oo}


def test_pset_inits():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset = Parameters([p1, p2, p3])
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
    pset = Parameters([p1, p2, p3])
    assert pset.nonfixed_inits == {'Y': 9, 'X': 3, 'Z': 1}
    pset['X'].fix = True
    assert pset.nonfixed_inits == {'Y': 9, 'Z': 1}


def test_pset_fix():
    p1 = Parameter('Y', 9, fix=False)
    p2 = Parameter('X', 3, fix=True)
    p3 = Parameter('Z', 1, fix=False)
    pset = Parameters([p1, p2, p3])
    assert pset.fix == {'Y': False, 'X': True, 'Z': False}
    fixedness = {'Y': True, 'X': True, 'Z': True}
    pset.fix = fixedness
    assert pset.fix == {'Y': True, 'X': True, 'Z': True}
    with pytest.raises(KeyError):
        pset.fix = {'K': True}


def test_pset_repr():
    p1 = Parameter('Y', 9, fix=False)
    pset = Parameters([p1])
    assert type(repr(pset)) == str
    assert type(pset._repr_html_()) == str
    pset = Parameters()
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
    pset1 = Parameters([p1, p2, p3])
    pset2 = Parameters([p1, p2])
    assert pset1 != pset2
    pset3 = Parameters([p1, p3, p2])
    assert pset1 != pset3
    assert pset1 == pset1


def test_pset_add():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset1 = Parameters([p1, p2])
    pset1.append(p3)
    assert len(pset1) == 3

    with pytest.raises(ValueError):
        pset1.append(23)


def test_pset_discard():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset1 = Parameters([p1, p2, p3])
    del pset1[p2]
    assert len(pset1) == 2
    del pset1['Y']
    assert len(pset1) == 1


def test_is_close_to_bound_pset():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3, lower=1, upper=24)
    p3 = Parameter('Z', 1, lower=0, upper=2)
    pset1 = Parameters([p1, p2, p3])
    assert not pset1.is_close_to_bound().any()
    assert not pset1.is_close_to_bound(pd.Series({'X': 3.5, 'Y': 19})).any()


def test_copy_pset():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3, lower=1, upper=24)
    p3 = Parameter('Z', 1, lower=0, upper=2)
    pset1 = Parameters([p1, p2, p3])
    pset2 = pset1.copy()
    assert pset1 == pset2
    assert id(pset1[0]) != id(pset2[0])
    p4 = p1.copy()
    assert p4 == p1
    assert id(p4) != id(p1)


def test_hash():
    p1 = Parameter('Y', 9)
    hash(p1)


def test_insert():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3, lower=1, upper=24)
    p3 = Parameter('Z', 1, lower=0, upper=2)
    pset1 = Parameters([p1, p2])
    pset1.insert(0, p3)
    assert pset1.names == ['Z', 'Y', 'X']

    p4 = Parameter('Y', 0)
    with pytest.raises(ValueError):
        pset1.insert(1, p4)


def test_simplify():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')

    p1 = Parameter('x', 3)
    p2 = Parameter('y', 9, fix=True)
    pset = Parameters([p1, p2])
    assert pset.simplify(x * y) == 9.0 * x

    p1 = Parameter('x', 3, lower=0.001)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    assert pset.simplify(abs(x)) == x

    p1 = Parameter('x', 3, lower=0)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    assert pset.simplify(sympy.Piecewise((2, sympy.Ge(x, 0)), (56, True))) == 2

    p1 = Parameter('x', -3, upper=-1)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    assert pset.simplify(abs(x)) == -x

    p1 = Parameter('x', -3, upper=0)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    assert pset.simplify(sympy.Piecewise((2, sympy.Le(x, 0)), (56, True))) == 2

    p1 = Parameter('x', 3)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    assert pset.simplify(x * y) == x * y


def test_unit():
    p = Parameter('x', 3, unit='l/h')
    assert p.unit == units.liter / units.hour
    p2 = Parameter('x', 3)
    assert p2.unit == 1
