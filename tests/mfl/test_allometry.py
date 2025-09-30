import pytest

from pharmpy.mfl.features import Allometry
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    a = Allometry(covariate='WT', reference=70.0)
    assert a.args == ('WT', 70.0)
    assert a.args == (a.covariate, a.reference)


def test_create():
    a1 = Allometry.create(covariate='wt')
    assert a1.args == ('WT', 70.0)
    assert a1.args == (a1.covariate, a1.reference)

    a2 = Allometry.create(covariate='wt', reference=80)
    assert a2.args == ('WT', 80.0)
    assert a2.args == (a2.covariate, a2.reference)

    with pytest.raises(TypeError):
        Allometry.create(1)

    with pytest.raises(TypeError):
        Allometry.create('wt', 'x')


def test_replace():
    a1 = Allometry.create(covariate='wt')
    assert a1.args == ('WT', 70.0)
    a2 = a1.replace(reference=80)
    assert a2.args == ('WT', 80.0)
    a3 = a2.replace(covariate='wgt')
    assert a3.args == ('WGT', 80.0)

    with pytest.raises(TypeError):
        a1.replace(covariate=1)

    with pytest.raises(TypeError):
        a1.replace(reference='x')


def test_repr():
    a1 = Allometry.create(covariate='WT', reference=70.0)
    assert repr(a1) == 'ALLOMETRY(WT,70)'
    a2 = Allometry.create(covariate='WT', reference=70.5)
    assert repr(a2) == 'ALLOMETRY(WT,70.5)'


def test_eq():
    a1 = Allometry.create(covariate='WT')
    assert a1 == a1
    a2 = Allometry.create(covariate='WT')
    assert a1 == a2
    a3 = Allometry.create(covariate='WT', reference=80)
    assert a1 != a3

    assert a1 != 1


def test_repr_many():
    a = Allometry.create(covariate='WT')
    mfl = ModelFeatures.create([a])
    assert Allometry.repr_many(mfl) == repr(a)
