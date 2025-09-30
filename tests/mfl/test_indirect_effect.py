import pytest

from pharmpy.mfl.features import IndirectEffect


def test_init():
    ie = IndirectEffect(type='LINEAR', production_type='DEGRADATION')
    assert ie.args == ('LINEAR', 'DEGRADATION')


def test_create():
    ie = IndirectEffect.create(type='linear', production_type='production')
    assert ie.args == ('LINEAR', 'PRODUCTION')

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
    assert IndirectEffect.repr_many([ie1]) == 'INDIRECTEFFECT(LINEAR,PRODUCTION)'
    ie2 = IndirectEffect.create(type='linear', production_type='degradation')
    assert IndirectEffect.repr_many([ie1, ie2]) == 'INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION])'
    ie3 = IndirectEffect.create(type='emax', production_type='production')
    assert (
        IndirectEffect.repr_many([ie3, ie2, ie1])
        == 'INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION]);INDIRECTEFFECT(EMAX,PRODUCTION)'
    )
    ie4 = IndirectEffect.create(type='emax', production_type='degradation')
    assert (
        IndirectEffect.repr_many([ie3, ie2, ie1, ie4])
        == 'INDIRECTEFFECT([LINEAR,EMAX],[DEGRADATION,PRODUCTION])'
    )
