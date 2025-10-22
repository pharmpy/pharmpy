import pytest

from pharmpy.mfl.features.transits import Transits
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    t1 = Transits(0, True)
    assert t1.args == (0, True)
    assert t1.args == (t1.number, t1.depot)


def test_create():
    t1 = Transits.create(0)
    assert t1.args == (0, True)
    assert t1.args == (t1.number, t1.depot)

    t2 = Transits.create(0, False)
    assert t2.args == (0, False)
    assert t2.args == (t2.number, t2.depot)

    t3 = Transits.create('n', False)
    assert t3.args == ('N', False)
    assert t3.args == (t3.number, t3.depot)

    with pytest.raises(ValueError):
        Transits.create('x')

    with pytest.raises(TypeError):
        Transits.create(1.0)

    with pytest.raises(ValueError):
        Transits.create(-1)

    with pytest.raises(TypeError):
        Transits.create(0, 1)


def test_replace():
    t1 = Transits.create(0)
    assert t1.args == (0, True)
    t2 = t1.replace(depot=False)
    assert t2.args == (0, False)
    t3 = t2.replace(number=1)
    assert t3.args == (1, False)

    with pytest.raises(TypeError):
        t1.replace(depot=1)

    with pytest.raises(ValueError):
        t1.replace(number='x')


def test_repr():
    t1 = Transits.create(0, True)
    assert repr(t1) == 'TRANSITS(0)'
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
    t4 = Transits.create('n')
    assert t3 < t4

    with pytest.raises(TypeError):
        t1 < 1


@pytest.mark.parametrize(
    'args, expected',
    (
        (
            ((0, True), (1, True), (2, True)),
            'TRANSITS([0,1,2])',
        ),
        (
            ((0, False), (1, False), (2, False)),
            'TRANSITS([0,1,2],NODEPOT)',
        ),
        (
            ((0, True), (1, True), (2, True), (0, False), (1, False), (2, False)),
            'TRANSITS([0,1,2],[DEPOT,NODEPOT])',
        ),
        (
            ((0, True), (1, True), (2, True), (0, False), (1, False)),
            'TRANSITS([0,1,2]);TRANSITS([0,1],NODEPOT)',
        ),
        (
            ((1, True),),
            'TRANSITS(1)',
        ),
        (
            ((0, True), (1, True), (2, True), ('N', True)),
            'TRANSITS(N);TRANSITS([0,1,2])',
        ),
        (
            ((0, False), (1, False), (2, False), ('N', True)),
            'TRANSITS(N);TRANSITS([0,1,2],NODEPOT)',
        ),
        (
            ((0, True), (1, True), (2, True), ('N', False)),
            'TRANSITS([0,1,2]);TRANSITS(N,NODEPOT)',
        ),
        (
            ((0, True), (1, True), (2, True), (0, False)),
            'TRANSITS([0,1,2]);TRANSITS(0,NODEPOT)',
        ),
    ),
)
def test_repr_many(args, expected):
    transits = [Transits.create(*arg) for arg in args]
    mf = ModelFeatures.create(transits)
    assert Transits.repr_many(mf) == expected
