import pytest

from pharmpy.mfl.features import Absorption, Elimination
from pharmpy.mfl.features.absorption import ABSORPTION_TYPES
from pharmpy.mfl.features.elimination import ELIMINATION_TYPES


@pytest.mark.parametrize(
    'feature_class, type',
    (
        (Absorption, 'FO'),
        (Elimination, 'FO'),
    ),
)
def test_init(feature_class, type):
    feature = feature_class(type)
    assert feature.args == (type,)


@pytest.mark.parametrize(
    'feature_class, type',
    (
        (Absorption, 'FO'),
        (Elimination, 'FO'),
    ),
)
def test_create(feature_class, type):
    feature = feature_class(type)
    assert feature.args == (type,)

    with pytest.raises(TypeError):
        feature_class.create(1)

    with pytest.raises(ValueError):
        feature_class.create('x')


@pytest.mark.parametrize(
    'feature_class, type, replace',
    (
        (Absorption, 'FO', 'ZO'),
        (Elimination, 'FO', 'MM'),
    ),
)
def test_replace(feature_class, type, replace):
    f1 = feature_class.create(type)
    assert f1.args == (type,)
    f2 = f1.replace(type=replace)
    assert f2.args == (replace,)

    with pytest.raises(TypeError):
        f1.replace(type=1)

    with pytest.raises(ValueError):
        f1.replace(type='x')


def test_repr():
    for abs_type in ABSORPTION_TYPES:
        assert str(Absorption.create(abs_type)) == f'ABSORPTION({abs_type})'
    for elim_type in ELIMINATION_TYPES:
        assert str(Elimination.create(elim_type)) == f'ELIMINATION({elim_type})'


@pytest.mark.parametrize(
    'feature_class, type_a, type_b',
    (
        (Absorption, 'FO', 'ZO'),
        (Elimination, 'FO', 'MM'),
    ),
)
def test_eq(feature_class, type_a, type_b):
    f1 = feature_class.create(type_a)
    assert f1 == f1
    f2 = feature_class.create(type_a)
    assert f1 == f2
    f3 = feature_class.create(type_b)
    assert f1 != f3

    assert f1 != 1


@pytest.mark.parametrize(
    'feature_class, types',
    (
        (Absorption, ABSORPTION_TYPES),
        (Elimination, ELIMINATION_TYPES),
    ),
)
def test_lt(feature_class, types):
    features = [feature_class.create(type) for type in types]
    order = list(feature_class.order.keys())

    features_sorted = sorted(features)
    assert all(order[i] in features_sorted[i].args for i in range(0, len(features_sorted)))

    with pytest.raises(TypeError):
        features_sorted[0] < 1


@pytest.mark.parametrize(
    'feature_class, types, expected',
    (
        (Absorption, ['FO', 'SEQ-ZO-FO', 'ZO'], 'ABSORPTION([FO,ZO,SEQ-ZO-FO])'),
        (Elimination, ['FO', 'MM', 'MIX-FO-MM'], 'ELIMINATION([FO,MM,MIX-FO-MM])'),
    ),
)
def test_repr_many(feature_class, types, expected):
    features = [feature_class.create(type) for type in types]
    assert feature_class.repr_many(features) == expected
    assert feature_class.repr_many([features[0]]) == f'{feature_class.__name__.upper()}({types[0]})'
