import pytest

from pharmpy.mfl.features.lagtime import LagTime
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    l1 = LagTime(on=True)
    assert l1.args == (True,)
    assert l1.args == (l1.on,)


def test_create():
    l1 = LagTime.create(on=True)
    assert l1.args == (True,)
    assert l1.args == (l1.on,)

    with pytest.raises(TypeError):
        LagTime.create(1)


def test_replace():
    l1 = LagTime.create(on=True)
    assert l1.args == (True,)
    l2 = l1.replace(on=False)
    assert l2.args == (False,)

    with pytest.raises(TypeError):
        l1.replace(on=1)


def test_repr():
    l1 = LagTime.create(on=True)
    assert repr(l1) == 'LAGTIME(ON)'
    l2 = LagTime.create(on=False)
    assert repr(l2) == 'LAGTIME(OFF)'


def test_eq():
    l1 = LagTime.create(on=True)
    assert l1 == l1
    l2 = LagTime.create(on=True)
    assert l1 == l2
    l3 = LagTime.create(on=False)
    assert l1 != l3

    assert l1 != 1


def test_lt():
    l1 = LagTime.create(on=True)
    assert not l1 < l1
    l2 = LagTime.create(on=False)
    assert l2 < l1
    with pytest.raises(TypeError):
        l1 < 1


def test_repr_many():
    l1 = LagTime.create(on=True)
    l2 = LagTime.create(on=False)
    mfl1 = ModelFeatures.create([l1])
    assert LagTime.repr_many(mfl1) == 'LAGTIME(ON)'
    mfl2 = mfl1 + l2
    assert LagTime.repr_many(mfl2) == 'LAGTIME([OFF,ON])'
