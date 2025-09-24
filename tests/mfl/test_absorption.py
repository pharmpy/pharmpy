import pytest

from pharmpy.mfl.features.absorption import ABSORPTION_TYPES, Absorption


def test_init():
    absorption = Absorption('FO')
    assert absorption.args == ('FO',)


def test_create():
    absorption = Absorption.create('FO')
    assert absorption.args == ('FO',)

    with pytest.raises(TypeError):
        Absorption.create(1)

    with pytest.raises(ValueError):
        Absorption.create('x')


def test_replace():
    absorption = Absorption.create('FO')
    assert absorption.args == ('FO',)
    absorption = absorption.replace(type='ZO')
    assert absorption.args == ('ZO',)

    with pytest.raises(TypeError):
        absorption.replace(type=1)

    with pytest.raises(ValueError):
        absorption.replace(type='x')


def test_expand():
    absorption = Absorption.create('FO')
    assert absorption.expand(None) == absorption


def test_repr():
    for abs_type in ABSORPTION_TYPES:
        assert str(Absorption.create(abs_type)) == f'ABSORPTION({abs_type})'


def test_eq():
    a1 = Absorption.create('FO')
    assert a1 == a1
    a2 = Absorption.create('FO')
    assert a1 == a2
    a3 = Absorption.create('ZO')
    assert a1 != a3

    assert a1 != 1


def test_lt():
    a1 = Absorption.create('FO')
    assert not a1 < a1
    a2 = Absorption.create('FO')
    assert not a1 < a2
    a3 = Absorption.create('ZO')
    assert a1 < a3
    a4 = Absorption.create('SEQ-ZO-FO')
    assert a3 < a4
    assert a1 < a4
    a5 = Absorption.create('WEIBULL')
    assert a4 < a5

    with pytest.raises(TypeError):
        a1 < 1


def test_repr_many():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    a3 = Absorption.create('SEQ-ZO-FO')
    assert Absorption.repr_many([a1, a2, a3]) == 'ABSORPTION([FO,ZO,SEQ-ZO-FO])'
    assert Absorption.repr_many([a1]) == 'ABSORPTION(FO)'
