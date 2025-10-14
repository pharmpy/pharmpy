import pytest

from pharmpy.mfl.features import IndirectEffect
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    ie = IndirectEffect(type='LINEAR', production=False)
    assert ie.args == ('LINEAR', False)
    assert ie.args == (ie.type, ie.production)


def test_create():
    ie = IndirectEffect.create(type='linear', production=True)
    assert ie.args == ('LINEAR', True)
    assert ie.args == (ie.type, ie.production)

    with pytest.raises(TypeError):
        IndirectEffect.create(1, True)

    with pytest.raises(TypeError):
        IndirectEffect.create('linear', 1)

    with pytest.raises(ValueError):
        IndirectEffect.create('x', True)


def test_replace():
    ie1 = IndirectEffect.create(type='linear', production=True)
    assert ie1.args == ('LINEAR', True)
    ie2 = ie1.replace(type='emax')
    assert ie2.args == ('EMAX', True)
    ie3 = ie2.replace(production=False)
    assert ie3.args == ('EMAX', False)

    with pytest.raises(TypeError):
        ie1.replace(type=1)

    with pytest.raises(TypeError):
        ie2.replace(production=1)


def test_repr():
    ie1 = IndirectEffect.create(type='linear', production=True)
    assert repr(ie1) == 'INDIRECTEFFECT(LINEAR,PRODUCTION)'
    ie2 = IndirectEffect.create(type='linear', production=False)
    assert repr(ie2) == 'INDIRECTEFFECT(LINEAR,DEGRADATION)'


def test_eq():
    ie1 = IndirectEffect.create(type='linear', production=True)
    assert ie1 == ie1
    ie2 = IndirectEffect.create(type='linear', production=True)
    assert ie1 == ie2
    ie3 = IndirectEffect.create(type='linear', production=False)
    assert ie1 != ie3

    assert ie1 != 1


def test_lt():
    ie1 = IndirectEffect.create(type='linear', production=True)
    assert not ie1 < ie1
    ie2 = IndirectEffect.create(type='linear', production=False)
    assert ie2 < ie1
    ie3 = IndirectEffect.create(type='emax', production=False)
    assert ie2 < ie3

    with pytest.raises(TypeError):
        ie1 < 1


def test_repr_many():
    ie1 = IndirectEffect.create(type='linear', production=True)
    mfl1 = ModelFeatures.create([ie1])
    assert IndirectEffect.repr_many(mfl1) == 'INDIRECTEFFECT(LINEAR,PRODUCTION)'
    ie2 = IndirectEffect.create(type='linear', production=False)
    mfl2 = mfl1 + ie2
    assert IndirectEffect.repr_many(mfl2) == 'INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION])'
    ie3 = IndirectEffect.create(type='emax', production=True)
    mfl3 = mfl2 + ie3
    assert (
        IndirectEffect.repr_many(mfl3)
        == 'INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION]);INDIRECTEFFECT(EMAX,PRODUCTION)'
    )
    ie4 = IndirectEffect.create(type='emax', production=False)
    mfl4 = mfl3 + ie4
    assert (
        IndirectEffect.repr_many(mfl4) == 'INDIRECTEFFECT([LINEAR,EMAX],[DEGRADATION,PRODUCTION])'
    )
