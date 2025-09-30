import pytest

from pharmpy.mfl.features.transits import Transits
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    t1 = Transits(0, True)
    assert t1.args == (0, True)
    assert t1.args == (t1.number, t1.with_depot)


def test_create():
    t1 = Transits.create(0)
    assert t1.args == (0, True)
    assert t1.args == (t1.number, t1.with_depot)

    t2 = Transits.create(0, False)
    assert t2.args == (0, False)
    assert t2.args == (t2.number, t2.with_depot)

    with pytest.raises(TypeError):
        Transits.create('x')

    with pytest.raises(ValueError):
        Transits.create(-1)

    with pytest.raises(TypeError):
        Transits.create(0, 1)


def test_replace():
    t1 = Transits.create(0)
    assert t1.args == (0, True)
    t2 = t1.replace(with_depot=False)
    assert t2.args == (0, False)
    t3 = t2.replace(number=1)
    assert t3.args == (1, False)

    with pytest.raises(TypeError):
        t1.replace(with_depot=1)

    with pytest.raises(TypeError):
        t1.replace(number='x')


def test_repr():
    t1 = Transits.create(0, True)
    assert repr(t1) == 'TRANSITS(0,DEPOT)'
    t2 = Transits.create(1, False)
    assert repr(t2) == 'TRANSITS(1,NODEPOT)'


def test_eq():
    t1 = Transits.create(0)
    assert t1 == t1
    t2 = Transits.create(0)
    assert t1 == t2
    t3 = Transits.create(0, False)
    assert t3 != t1
    t4 = Transits.create(1)
    assert t4 != t1

    assert t1 != 1


def test_lt():
    t1 = Transits.create(0)
    assert not t1 < t1
    t2 = Transits.create(0, False)
    assert t1 < t2
    t3 = Transits.create(1)
    assert t1 < t3
    assert t3 < t2

    with pytest.raises(TypeError):
        t1 < 1


def test_repr_many():
    t_depot = [Transits.create(i) for i in range(3)]
    mfl1 = ModelFeatures.create(t_depot)
    assert Transits.repr_many(mfl1) == 'TRANSITS([0,1,2],DEPOT)'
    t_nodepot = [Transits.create(i, False) for i in range(1, 3)]
    mfl2 = ModelFeatures.create(t_nodepot)
    assert Transits.repr_many(mfl2) == 'TRANSITS([1,2],NODEPOT)'
    assert Transits.repr_many(mfl1 + mfl2) == 'TRANSITS([0,1,2],DEPOT);TRANSITS([1,2],NODEPOT)'
    mfl3 = ModelFeatures.create([t_depot[0]])
    assert Transits.repr_many(mfl3) == 'TRANSITS(0,DEPOT)'
    assert Transits.repr_many(mfl3 + mfl2) == 'TRANSITS(0,DEPOT);TRANSITS([1,2],NODEPOT)'
