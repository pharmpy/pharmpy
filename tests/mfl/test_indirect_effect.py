import pytest

from pharmpy.mfl.features import IndirectEffect
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    ie = IndirectEffect(type='LINEAR', production_type='DEGRADATION')
    assert ie.args == ('LINEAR', 'DEGRADATION')
    assert ie.args == (ie.type, ie.production_type)


def test_create():
    ie = IndirectEffect.create(type='linear', production_type='production')
    assert ie.args == ('LINEAR', 'PRODUCTION')
    assert ie.args == (ie.type, ie.production_type)

    with pytest.raises(TypeError):
        IndirectEffect.create(1, 'production')

    with pytest.raises(TypeError):
        IndirectEffect.create('linear', 1)

    with pytest.raises(ValueError):
        IndirectEffect.create('x', 'production')
    with pytest.raises(ValueError):
        IndirectEffect.create('linear', 'x')


def test_replace():
    ie1 = IndirectEffect.create(type='linear', production_type='production')
    assert ie1.args == ('LINEAR', 'PRODUCTION')
    ie2 = ie1.replace(type='emax')
    assert ie2.args == ('EMAX', 'PRODUCTION')
    ie3 = ie2.replace(production_type='degradation')
    assert ie3.args == ('EMAX', 'DEGRADATION')

    with pytest.raises(TypeError):
        ie1.replace(type=1)

    with pytest.raises(TypeError):
        ie2.replace(production_type=1)


def test_repr():
    ie = IndirectEffect.create(type='linear', production_type='production')
    assert repr(ie) == 'INDIRECTEFFECT(LINEAR,PRODUCTION)'


def test_eq():
    ie1 = IndirectEffect.create(type='linear', production_type='production')
    assert ie1 == ie1
    ie2 = IndirectEffect.create(type='linear', production_type='production')
    assert ie1 == ie2
    ie3 = IndirectEffect.create(type='linear', production_type='degradation')
    assert ie1 != ie3

    assert ie1 != 1


def test_lt():
    ie1 = IndirectEffect.create(type='linear', production_type='production')
    assert not ie1 < ie1
    ie2 = IndirectEffect.create(type='linear', production_type='degradation')
    assert ie2 < ie1
    ie3 = IndirectEffect.create(type='emax', production_type='degradation')
    assert ie2 < ie3

    with pytest.raises(TypeError):
        ie1 < 1


def test_repr_many():
    ie1 = IndirectEffect.create(type='linear', production_type='production')
    mfl1 = ModelFeatures.create([ie1])
    assert IndirectEffect.repr_many(mfl1) == 'INDIRECTEFFECT(LINEAR,PRODUCTION)'
    ie2 = IndirectEffect.create(type='linear', production_type='degradation')
    mfl2 = mfl1 + ie2
    assert IndirectEffect.repr_many(mfl2) == 'INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION])'
    ie3 = IndirectEffect.create(type='emax', production_type='production')
    mfl3 = mfl2 + ie3
    assert (
        IndirectEffect.repr_many(mfl3)
        == 'INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION]);INDIRECTEFFECT(EMAX,PRODUCTION)'
    )
    ie4 = IndirectEffect.create(type='emax', production_type='degradation')
    mfl4 = mfl3 + ie4
    assert (
        IndirectEffect.repr_many(mfl4) == 'INDIRECTEFFECT([LINEAR,EMAX],[DEGRADATION,PRODUCTION])'
    )
