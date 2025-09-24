import pytest

from pharmpy.mfl.features import Absorption, ModelFeature, Peripherals, Transits
from pharmpy.mfl.model_features import ModelFeatures


class ModelFeatureX(ModelFeature):
    @property
    def args(self):
        return tuple()

    def expand(self, model):
        return self


def test_init():
    mf1 = ModelFeatures(tuple())
    assert mf1.features == tuple()
    a = Absorption.create('FO')
    mf2 = ModelFeatures((a,))
    assert mf2.features == (a,)


def test_create():
    a1 = Absorption.create('FO')
    mf1 = ModelFeatures.create([a1])
    assert mf1.features == (a1,)
    assert len(mf1) == 1

    a2 = Absorption.create('ZO')
    mf2 = ModelFeatures.create([a1, a2])
    assert mf2.features == (a1, a2)
    assert len(mf2) == 2

    mf3 = ModelFeatures.create([a1, a2, a1])
    assert mf3.features == (a1, a2)
    assert len(mf3) == 2

    p1 = Peripherals.create(0)
    mf4 = ModelFeatures.create([a1, p1, a2])
    assert mf4.features == (a1, a2, p1)

    t1 = Transits.create(0)
    mf5 = ModelFeatures.create([a1, p1, t1, a2])
    assert mf5.features == (a1, a2, t1, p1)

    with pytest.raises(TypeError):
        ModelFeatures.create(features=1)

    with pytest.raises(TypeError):
        ModelFeatures.create(features=[1])

    with pytest.raises(TypeError):
        ModelFeatures.create(features=[a1, 1])

    x = ModelFeatureX()

    with pytest.raises(NotImplementedError):
        ModelFeatures.create(features=[x])


def test_replace():
    a1 = Absorption.create('FO')
    mf1 = ModelFeatures.create([a1])
    assert mf1.features == (a1,)

    a2 = Absorption.create('ZO')
    mf2 = mf1.replace(features=[a1, a2])
    assert mf2.features == (a1, a2)

    mf3 = mf2.replace(features=[a1])
    assert mf3.features == (a1,)

    with pytest.raises(TypeError):
        mf1.replace(features=[1])

    with pytest.raises(TypeError):
        mf1.replace(features=[a1, 1])


def test_absorption():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    mf1 = ModelFeatures.create([a1, a2])
    assert mf1.features == (a1, a2)
    assert mf1.absorption.features == (a1, a2)

    x = ModelFeatureX()
    mf2 = ModelFeatures((a1, a2, x))
    assert mf2.features == (a1, a2, x)
    assert mf2.absorption.features == (a1, a2)


def test_transits():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    t1 = Transits.create(0)
    t2 = Transits.create(0, False)
    mf = ModelFeatures.create([a1, t2, a2, t1])
    assert mf.features == (a1, a2, t1, t2)
    assert mf.transits.features == (t1, t2)


def test_peripherals():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    p1 = Peripherals.create(0)
    p2 = Peripherals.create(0, 'MET')
    mf = ModelFeatures.create([a1, p2, a2, p1])
    assert mf.features == (a1, a2, p1, p2)
    assert mf.peripherals.features == (p1, p2)


def test_add():
    a1 = Absorption.create('FO')
    mf1 = ModelFeatures.create([a1])

    a2 = Absorption.create('ZO')
    mf2 = mf1 + a2
    assert mf2.features == (a1, a2)
    assert a2 + mf1 == mf2

    a3 = Absorption.create('SEQ-ZO-FO')
    mf3 = ModelFeatures.create([a3])
    mf4 = mf2 + mf3
    assert mf4.features == (a1, a2, a3)

    mf5 = mf1 + [a2, a3]
    assert mf5.features == (a1, a2, a3)
    assert mf5.features == (a1, a2, a3)

    with pytest.raises(TypeError):
        mf1 + 1


def test_eq():
    a1 = Absorption.create('FO')
    mf1 = ModelFeatures.create([a1])
    assert mf1 == mf1

    a2 = Absorption.create('ZO')
    mf2 = ModelFeatures.create([a1, a2])
    assert mf1 != mf2

    a3 = Absorption.create('SEQ-ZO-FO')
    mf3 = ModelFeatures.create([a1, a3])
    assert mf2 != mf3

    mf4 = ModelFeatures.create([a3, a1])
    assert mf3 == mf4

    assert mf1 != 1


@pytest.mark.parametrize(
    'features, expected',
    (
        (
            [Absorption.create('FO'), Absorption.create('SEQ-ZO-FO'), Absorption.create('ZO')],
            'ABSORPTION([FO,ZO,SEQ-ZO-FO])',
        ),
        (
            [
                Absorption.create('FO'),
                Peripherals.create(0),
                Peripherals.create(1),
                Peripherals.create(2),
                Absorption.create('ZO'),
            ],
            'ABSORPTION([FO,ZO]);PERIPHERALS(0..2)',
        ),
        (
            [
                Absorption.create('FO'),
                Peripherals.create(0),
                Peripherals.create(1),
                Peripherals.create(0, 'MET'),
                Absorption.create('ZO'),
            ],
            'ABSORPTION([FO,ZO]);PERIPHERALS(0..1);PERIPHERALS(0,MET)',
        ),
        (
            [
                Peripherals.create(0),
                Peripherals.create(1),
                Peripherals.create(0, 'MET'),
                Transits.create(0),
                Transits.create(1),
                Transits.create(3),
                Transits.create(0, False),
                Transits.create(1, False),
                Transits.create(3, False),
            ],
            'TRANSITS([0,1,3],DEPOT);TRANSITS([0,1,3],NODEPOT);PERIPHERALS(0..1);PERIPHERALS(0,MET)',
        ),
    ),
)
def test_repr(features, expected):
    mf = ModelFeatures.create(features)
    assert repr(mf) == expected
