import pytest

from pharmpy.mfl.features.peripherals import Peripherals
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    p1 = Peripherals(0, metabolite=False)
    assert p1.args == (0, False)
    assert p1.args == (p1.number, p1.metabolite)


def test_create():
    p1 = Peripherals.create(0)
    assert p1.args == (0, False)
    assert p1.args == (p1.number, p1.metabolite)

    p2 = Peripherals.create(0, metabolite=True)
    assert p2.args == (0, True)
    assert p2.args == (p2.number, p2.metabolite)

    with pytest.raises(TypeError):
        Peripherals.create('x')

    with pytest.raises(ValueError):
        Peripherals.create(-1)

    with pytest.raises(TypeError):
        Peripherals.create(0, 1)


def test_replace():
    p1 = Peripherals.create(0)
    assert p1.args == (0, False)
    p2 = p1.replace(metabolite=True)
    assert p2.args == (0, True)
    p3 = p2.replace(number=1)
    assert p3.args == (1, True)

    with pytest.raises(TypeError):
        p1.replace(metabolite=1)

    with pytest.raises(TypeError):
        p1.replace(number='x')


def test_repr():
    p1 = Peripherals.create(0, metabolite=False)
    assert repr(p1) == 'PERIPHERALS(0)'
    p2 = Peripherals.create(0, metabolite=True)
    assert repr(p2) == 'PERIPHERALS(0,MET)'


def test_eq():
    p1 = Peripherals.create(0)
    assert p1 == p1
    p2 = Peripherals.create(0)
    assert p1 == p2
    p3 = Peripherals.create(0, True)
    assert p3 != p1
    p4 = Peripherals.create(1)
    assert p4 != p1

    assert p1 != 1


def test_lt():
    p1 = Peripherals.create(0)
    assert not p1 < p1
    p2 = Peripherals.create(0, True)
    assert p1 < p2
    p3 = Peripherals.create(1)
    assert p1 < p3
    assert p3 < p2

    with pytest.raises(TypeError):
        p1 < 1


def test_repr_many():
    p_drug = [Peripherals.create(i) for i in range(3)]
    mfl1 = ModelFeatures.create(p_drug)
    assert Peripherals.repr_many(mfl1) == 'PERIPHERALS(0..2)'
    p_met = [Peripherals.create(i, True) for i in range(1, 3)]
    mfl2 = ModelFeatures.create(p_met)
    assert Peripherals.repr_many(mfl2) == 'PERIPHERALS(1..2,MET)'
    assert Peripherals.repr_many(mfl1 + mfl2) == 'PERIPHERALS(0..2);PERIPHERALS(1..2,MET)'
    p_non_consec = [Peripherals.create(i) for i in [0, 2, 5]]
    mfl3 = ModelFeatures.create(p_non_consec)
    assert Peripherals.repr_many(mfl3) == 'PERIPHERALS([0,2,5])'
    mfl4 = ModelFeatures.create([p_drug[0]])
    assert Peripherals.repr_many(mfl4) == 'PERIPHERALS(0)'
