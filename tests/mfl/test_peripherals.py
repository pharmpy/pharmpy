import pytest

from pharmpy.mfl.features.peripherals import Peripherals


def test_init():
    absorption = Peripherals(0, 'DRUG')
    assert absorption.args == (0, 'DRUG')


def test_create():
    p1 = Peripherals.create(0)
    assert p1.args == (0, 'DRUG')

    p2 = Peripherals.create(0, 'met')
    assert p2.args == (0, 'MET')

    with pytest.raises(TypeError):
        Peripherals.create('x')

    with pytest.raises(ValueError):
        Peripherals.create(-1)

    with pytest.raises(TypeError):
        Peripherals.create(0, 1)

    with pytest.raises(ValueError):
        Peripherals.create(0, 'x')


def test_replace():
    p1 = Peripherals.create(0)
    assert p1.args == (0, 'DRUG')
    p2 = p1.replace(type='met')
    assert p2.args == (0, 'MET')
    p3 = p2.replace(number=1)
    assert p3.args == (1, 'MET')

    with pytest.raises(TypeError):
        p1.replace(type=1)

    with pytest.raises(TypeError):
        p1.replace(number='x')


def test_expand():
    p1 = Peripherals.create(0)
    assert p1.expand(None) == p1


def test_repr():
    p1 = Peripherals.create(0, 'DRUG')
    assert repr(p1) == 'PERIPHERALS(0)'
    p2 = Peripherals.create(0, 'MET')
    assert repr(p2) == 'PERIPHERALS(0,MET)'


def test_eq():
    p1 = Peripherals.create(0)
    assert p1 == p1
    p2 = Peripherals.create(0)
    assert p1 == p2
    p3 = Peripherals.create(0, 'MET')
    assert p3 != p1
    p4 = Peripherals.create(1)
    assert p4 != p1

    assert p1 != 1


def test_lt():
    p1 = Peripherals.create(0)
    assert not p1 < p1
    p2 = Peripherals.create(0, 'MET')
    assert p1 < p2
    p3 = Peripherals.create(1)
    assert p1 < p3
    assert p3 < p2

    with pytest.raises(TypeError):
        p1 < 1


def test_repr_many():
    p_drug = [Peripherals.create(i) for i in range(3)]
    assert Peripherals.repr_many(p_drug) == 'PERIPHERALS(0..2)'
    p_met = [Peripherals.create(i, 'MET') for i in range(1, 3)]
    assert Peripherals.repr_many(p_met) == 'PERIPHERALS(1..2,MET)'
    assert Peripherals.repr_many(p_drug + p_met) == 'PERIPHERALS(0..2);PERIPHERALS(1..2,MET)'
    p_non_consec = [Peripherals.create(i) for i in [0, 2, 5]]
    assert Peripherals.repr_many(p_non_consec) == 'PERIPHERALS([0,2,5])'
    assert Peripherals.repr_many([]) == ''
    p_one = Peripherals.create(0)
    assert Peripherals.repr_many([p_one]) == 'PERIPHERALS(0)'
