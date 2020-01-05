import pytest
import sympy

from pharmpy.parameter import Parameter, ParameterSet


@pytest.mark.parametrize('name,init,lower,upper,fix', [
    ('THETA(1)', 23, None, None, None),
    ('X', 12, None, None, None),
    ('_NAME', 0, None, None, None),
    ('OMEGA(2,1)', 0.1, 0, None, None),
    ('TCVL', 0.23, -2, 2, None),
    ])
def test_initialization(name, init, lower, upper, fix):
    param = Parameter(name, init, lower, upper, fix)
    assert param.symbol.name == name
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


@pytest.mark.parametrize('name,init,lower,upper,fix', [
    ('OMEGA(2,1)', 0.1, 2, None, None),
    ('X', 1, 0, -1, None),
    ('X', 1, 0, 1, True),
    ])
def test_illegal_initialization(name, init, lower, upper, fix):
    with pytest.raises(ValueError):
        Parameter(name, init, lower, upper, fix)


def test_unconstrain():
    param = Parameter('X', 2, lower=0, upper=23)
    param.unconstrain()
    assert param.lower == -sympy.oo
    assert param.upper == sympy.oo

    fixed_param = Parameter('Y', 0, fix=True)
    fixed_param.unconstrain()
    assert fixed_param.lower == -sympy.oo
    assert fixed_param.upper == sympy.oo


def test_pset_index():
    p = Parameter('Y', 9)
    pset = ParameterSet((p,))
    assert len(pset) == 1
    assert pset['Y'] is p

    p2 = Parameter('Z', 5)
    pset.add(p2)

    assert len(pset) == 2
    assert pset[p2.symbol] is p2

    # Check that the parameter set keeps the insertion order upon iteration
    for i, param in enumerate(pset):
        if i == 0:
            assert param is p
        else:
            assert param is p2
