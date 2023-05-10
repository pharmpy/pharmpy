import numpy as np
import pytest
import sympy
from sympy import Symbol as symbol

from pharmpy.model import Parameter, Parameters


@pytest.mark.parametrize(
    'name,init,lower,upper,fix',
    [
        ('THETA(1)', 23, None, None, None),
        ('X', 12, None, None, None),
        ('_NAME', 0, None, None, False),
        ('OMEGA(2,1)', 0.1, 0, None, None),
        ('TCVL', 0.23, -2, 2, None),
    ],
)
def test_initialization(name, init, lower, upper, fix):
    if lower is None:
        lower = -sympy.oo
    if upper is None:
        upper = sympy.oo
    if fix is None:
        fix = False
    param = Parameter(name, init, lower, upper, fix)
    assert param.name == name
    assert param.symbol == symbol(name)
    assert param.init == init
    assert param.lower == lower
    assert param.upper == upper
    assert param.fix == bool(fix)


@pytest.mark.parametrize(
    'name,init,lower,upper,fix',
    [
        ('OMEGA(2,1)', 0.1, 2, None, None),
        ('X', 1, 0, -1, None),
        ('X', np.nan, 0, 2, None),
        (23, 0, None, None, None),
    ],
)
def test_illegal_initialization(name, init, lower, upper, fix):
    with pytest.raises(ValueError):
        Parameter.create(name, init, lower, upper, fix)


def test_repr():
    param = Parameter('X', 2, lower=0, upper=23)
    assert repr(param) == 'Parameter("X", 2, lower=0, upper=23, fix=False)'
    param = Parameter('X', 2)
    assert repr(param) == 'Parameter("X", 2, lower=-∞, upper=∞, fix=False)'


def test_fix():
    param = Parameter('X', 2, lower=0, upper=23)
    assert param == Parameter('X', 2, lower=0, upper=23, fix=False)
    param = param.replace(fix=True)
    assert param == Parameter('X', 2, lower=0, upper=23, fix=True)


def test_add():
    p1 = Parameter.create('x', 2)
    p2 = Parameter.create('y', 3)
    p3 = Parameter.create('z', 4)
    pset = Parameters((p1, p2))
    pset2 = pset + p3
    assert len(pset2) == 3
    assert pset2['z'].init == 4
    pset3 = Parameters((p1,))
    pset4 = pset3 + Parameters((p2, p3))
    assert len(pset4) == 3
    assert pset4['z'].init == 4
    pset5 = Parameters((p1,))
    pset6 = pset5 + (p2, p3)
    assert len(pset6) == 3
    assert pset6['z'].init == 4

    with pytest.raises(ValueError):
        pset + 23


def test_pset_radd():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    p4 = Parameter.create('W', 1)
    cat = p4 + pset1
    assert len(cat) == 4
    assert cat['W'].init == 1
    assert cat['Y'].init == 9

    cat = [p4] + pset1
    assert len(cat) == 4
    assert cat['W'].init == 1
    assert cat['Y'].init == 9

    with pytest.raises(ValueError):
        23 + pset1


def test_pset_init():
    p = Parameter.create('Y', 9)
    pset = Parameters((p,))
    pset2 = Parameters.create(pset)
    assert len(pset2) == 1
    assert pset2['Y'].init == 9

    p2 = Parameter.create('Y', 12)
    with pytest.raises(ValueError):
        Parameters.create((p, p2))

    with pytest.raises(ValueError):
        Parameters.create([23])

    pset3 = Parameters.create()
    assert len(pset3) == 0


def test_pset_getitem():
    p = Parameter.create('Y', 9)
    pset = Parameters((p,))
    assert len(pset) == 1
    assert pset['Y'] is p

    p2 = Parameter.create('Z', 5)
    pset = Parameters((p, p2))

    assert len(pset) == 2

    # Check that the parameter set keeps the insertion order upon iteration
    for i, param in enumerate(pset):
        if i == 0:
            assert param is p
        else:
            assert param is p2

    assert pset[symbol('Z')] == p2

    p3 = Parameter.create('K', 19)
    with pytest.raises(KeyError):
        pset[p3]

    assert len(pset[[p]]) == 1

    with pytest.raises(KeyError):
        pset['noparamofmine']


def test_pset_nonfixed():
    p1 = Parameter.create('Y', 9, fix=False)
    p2 = Parameter.create('X', 3, fix=True)
    p3 = Parameter.create('Z', 1, fix=False)
    pset = Parameters((p1, p2, p3))
    assert len(pset.nonfixed) == 2
    assert pset.nonfixed['Y'] == Parameter.create('Y', 9)


def test_pset_names():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset = Parameters((p1, p2, p3))
    assert pset.names == ['Y', 'X', 'Z']
    assert pset.symbols == [symbol('Y'), symbol('X'), symbol('Z')]


def test_pset_lower_upper():
    p1 = Parameter('X', 0, lower=-1, upper=1)
    p2 = Parameter.create('Y', 1, lower=0)
    pset = Parameters((p1, p2))
    assert pset.lower == {'X': -1, 'Y': 0}
    assert pset.upper == {'X': 1, 'Y': sympy.oo}


def test_pset_nonfixed_inits():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset = Parameters((p1, p2, p3))
    assert pset.nonfixed.inits == {'Y': 9, 'X': 3, 'Z': 1}


def test_pset_fix():
    p1 = Parameter.create('Y', 9, fix=False)
    p2 = Parameter.create('X', 3, fix=True)
    p3 = Parameter.create('Z', 1, fix=False)
    pset = Parameters((p1, p2, p3))
    assert pset.fix == {'Y': False, 'X': True, 'Z': False}


def test_pset_repr():
    p1 = Parameter.create('Y', 9, fix=False)
    pset = Parameters((p1,))
    assert type(repr(pset)) == str
    assert type(pset._repr_html_()) == str
    pset = Parameters()
    assert type(repr(pset)) == str
    assert type(pset._repr_html_()) == str


def test_pset_eq():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    pset2 = Parameters((p1, p2))
    assert pset1 != pset2
    pset3 = Parameters((p1, p3, p2))
    assert pset1 != pset3
    assert pset1 == pset1
    assert pset1 != 23


def test_pset_replace():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    pset2 = pset1.replace(parameters=(p1, p2))
    assert len(pset2) == 2


def test_hash():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('Y', 9, upper=23)
    assert hash(p1) != hash(p2)
    p3 = Parameter.create('X', 9)
    p4 = Parameter.create('Z', 9)
    pset1 = Parameters((p1, p3))
    pset2 = Parameters((p1, p4))
    assert hash(pset1) != hash(pset2)


def test_dict():
    p1 = Parameter.create('Y', 9)
    d = p1.to_dict()
    assert d == {
        'name': 'Y',
        'init': 9.0,
        'lower': -float('inf'),
        'upper': float('inf'),
        'fix': False,
    }
    p2 = Parameter.from_dict(d)
    assert p1 == p2
    p3 = Parameter.create('Z', 0, lower=0, upper=22)
    pset = Parameters((p1, p3))
    d2 = pset.to_dict()
    assert d2 == {
        'parameters': (
            {'name': 'Y', 'init': 9.0, 'lower': -float('inf'), 'upper': float('inf'), 'fix': False},
            {'name': 'Z', 'init': 0.0, 'lower': 0.0, 'upper': 22.0, 'fix': False},
        )
    }
    pset2 = Parameters.from_dict(d2)
    assert pset == pset2


def test_contains():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    assert 'Y' in pset1
    assert 'Q' not in pset1


def test_set_initial_estimates():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    pset2 = pset1.set_initial_estimates({'Y': 23, 'Z': 19})
    assert pset2['Y'].init == 23
    assert pset2['X'].init == 3
    assert pset2['Z'].init == 19
    assert pset1['Y'].init == 9
    assert pset1['X'].init == 3
    assert pset1['Z'].init == 1


def test_set_fix():
    p1 = Parameter.create('Y', 9, fix=True)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    pset2 = pset1.set_fix({'Y': False, 'X': True})
    assert not pset2['Y'].fix
    assert pset2['X'].fix
    assert not pset2['Z'].fix
    assert pset1['Y'].fix
    assert not pset1['X'].fix
    assert not pset1['Z'].fix


def test_fixed_nonfixed():
    p1 = Parameter.create('Y', 9, fix=True)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    pset_fixed = Parameters((p1,))
    pset_nonfixed = Parameters((p2, p3))
    assert pset1.fixed == pset_fixed
    assert pset1.nonfixed == pset_nonfixed


def test_slice():
    p1 = Parameter.create('Y', 9)
    p2 = Parameter.create('X', 3)
    p3 = Parameter.create('Z', 1)
    pset1 = Parameters((p1, p2, p3))
    sl = pset1[1:]
    assert len(sl) == 2
    assert sl['X'].init == 3
    assert 'Y' not in sl


def test_replace():
    p = Parameter.create('x', 3, fix=False)
    p1 = p.replace(upper=4)
    assert not p1.fix
    assert not p.fix
    assert p1.name == 'x'
    assert p1.upper == 4

    p2 = p.replace(lower=2)
    assert p2.lower == 2

    p3 = p.replace(init=23)
    assert p3.init == 23
