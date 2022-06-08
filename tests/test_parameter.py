import numpy as np
import pytest
import sympy

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
        ('X', np.nan, 0, 2, None),
        (23, 0, None, None, None),
    ],
)
def test_illegal_initialization(name, init, lower, upper, fix):
    with pytest.raises(ValueError):
        Parameter(name, init, lower, upper, fix)


def test_repr():
    param = Parameter('X', 2, lower=0, upper=23)
    assert repr(param) == 'Parameter("X", 2, lower=0, upper=23, fix=False)'


def test_fix():
    param = Parameter('X', 2, lower=0, upper=23)
    assert param == Parameter('X', 2, lower=0, upper=23, fix=False)
    param = param.set_fix(True)
    assert param == Parameter('X', 2, lower=0, upper=23, fix=True)


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
    pset = Parameters((p, p2))

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

    with pytest.raises(KeyError):
        pset['noparamofmine']


def test_pset_nonfixed():
    p1 = Parameter('Y', 9, fix=False)
    p2 = Parameter('X', 3, fix=True)
    p3 = Parameter('Z', 1, fix=False)
    pset = Parameters([p1, p2, p3])
    assert len(pset.nonfixed) == 2
    assert pset.nonfixed['Y'] == Parameter('Y', 9)


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


def test_pset_nonfixed_inits():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset = Parameters([p1, p2, p3])
    assert pset.nonfixed.inits == {'Y': 9, 'X': 3, 'Z': 1}


def test_pset_fix():
    p1 = Parameter('Y', 9, fix=False)
    p2 = Parameter('X', 3, fix=True)
    p3 = Parameter('Z', 1, fix=False)
    pset = Parameters([p1, p2, p3])
    assert pset.fix == {'Y': False, 'X': True, 'Z': False}


def test_pset_repr():
    p1 = Parameter('Y', 9, fix=False)
    pset = Parameters([p1])
    assert type(repr(pset)) == str
    assert type(pset._repr_html_()) == str
    pset = Parameters()
    assert type(repr(pset)) == str
    assert type(pset._repr_html_()) == str


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


def test_hash():
    p1 = Parameter('Y', 9)
    hash(p1)


def test_contains():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset1 = Parameters([p1, p2, p3])
    assert 'Y' in pset1
    assert 'Q' not in pset1


def test_set_initial_estimates():
    p1 = Parameter('Y', 9)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset1 = Parameters([p1, p2, p3])
    pset2 = pset1.set_initial_estimates({'Y': 23, 'Z': 19})
    assert pset2['Y'].init == 23
    assert pset2['X'].init == 3
    assert pset2['Z'].init == 19
    assert pset1['Y'].init == 9
    assert pset1['X'].init == 3
    assert pset1['Z'].init == 1


def test_set_fix():
    p1 = Parameter('Y', 9, fix=True)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset1 = Parameters([p1, p2, p3])
    pset2 = pset1.set_fix({'Y': False, 'X': True})
    assert not pset2['Y'].fix
    assert pset2['X'].fix
    assert not pset2['Z'].fix
    assert pset1['Y'].fix
    assert not pset1['X'].fix
    assert not pset1['Z'].fix


def test_fixed_nonfixed():
    p1 = Parameter('Y', 9, fix=True)
    p2 = Parameter('X', 3)
    p3 = Parameter('Z', 1)
    pset1 = Parameters([p1, p2, p3])
    pset_fixed = Parameters([p1])
    pset_nonfixed = Parameters([p2, p3])
    assert pset1.fixed == pset_fixed
    assert pset1.nonfixed == pset_nonfixed
