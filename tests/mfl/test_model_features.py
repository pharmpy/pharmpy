import pytest

from pharmpy.mfl.absorption import Absorption
from pharmpy.mfl.model_feature import ModelFeature
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


def test_repr():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('SEQ-ZO-FO')
    a3 = Absorption.create('ZO')
    mf = ModelFeatures.create([a1, a2, a3])
    assert repr(mf) == 'ABSORPTION([FO,ZO,SEQ-ZO-FO])'
