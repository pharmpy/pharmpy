import pytest

from pharmpy.mfl.features import Absorption, DirectEffect, EffectComp, Elimination, Metabolite
from pharmpy.mfl.features.absorption import ABSORPTION_TYPES
from pharmpy.mfl.features.direct_effect import DIRECT_EFFECT_TYPES
from pharmpy.mfl.features.effect_compartment import EFFECT_COMP_TYPES
from pharmpy.mfl.features.elimination import ELIMINATION_TYPES
from pharmpy.mfl.features.metabolite import METABOLITE_TYPES
from pharmpy.mfl.model_features import ModelFeatures


@pytest.mark.parametrize(
    'feature_class, type',
    (
        (Absorption, 'FO'),
        (Elimination, 'FO'),
        (DirectEffect, 'LINEAR'),
        (EffectComp, 'LINEAR'),
        (Metabolite, 'PSC'),
    ),
)
def test_init(feature_class, type):
    feature = feature_class(type)
    assert feature.args == (type,)
    assert feature.args == (feature.type,)


@pytest.mark.parametrize(
    'feature_class, type',
    (
        (Absorption, 'fo'),
        (Elimination, 'fo'),
        (DirectEffect, 'linear'),
        (EffectComp, 'linear'),
        (Metabolite, 'psc'),
    ),
)
def test_create(feature_class, type):
    feature = feature_class.create(type)
    assert feature.args == (type.upper(),)
    assert feature.args == (feature.type,)

    with pytest.raises(TypeError):
        feature_class.create(1)

    with pytest.raises(ValueError):
        feature_class.create('x')


@pytest.mark.parametrize(
    'feature_class, type, replace',
    (
        (Absorption, 'FO', 'ZO'),
        (Elimination, 'FO', 'MM'),
        (DirectEffect, 'LINEAR', 'SIGMOID'),
        (EffectComp, 'LINEAR', 'SIGMOID'),
        (Metabolite, 'PSC', 'BASIC'),
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
    for type in ABSORPTION_TYPES:
        assert str(Absorption.create(type)) == f'ABSORPTION({type})'
    for type in ELIMINATION_TYPES:
        assert str(Elimination.create(type)) == f'ELIMINATION({type})'
    for type in DIRECT_EFFECT_TYPES:
        assert str(DirectEffect.create(type)) == f'DIRECTEFFECT({type})'
    for type in EFFECT_COMP_TYPES:
        assert str(EffectComp.create(type)) == f'EFFECTCOMP({type})'
    for type in METABOLITE_TYPES:
        assert str(Metabolite.create(type)) == f'METABOLITE({type})'


@pytest.mark.parametrize(
    'feature_class, type_a, type_b',
    (
        (Absorption, 'FO', 'ZO'),
        (Elimination, 'FO', 'MM'),
        (DirectEffect, 'LINEAR', 'SIGMOID'),
        (EffectComp, 'LINEAR', 'SIGMOID'),
        (Metabolite, 'PSC', 'BASIC'),
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
    'feature_class, order, types',
    (
        (Absorption, ('FO', 'ZO', 'SEQ-ZO-FO', 'WEIBULL'), ABSORPTION_TYPES),
        (Elimination, ('FO', 'ZO', 'MM', 'MIX-FO-MM'), ELIMINATION_TYPES),
        (DirectEffect, ('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN'), DIRECT_EFFECT_TYPES),
        (EffectComp, ('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN'), EFFECT_COMP_TYPES),
        (Metabolite, ('PSC', 'BASIC'), METABOLITE_TYPES),
    ),
)
def test_lt(feature_class, order, types):
    features = [feature_class.create(type) for type in types]
    features_sorted = sorted(features)
    assert all(order[i] in features_sorted[i].args for i in range(0, len(features_sorted)))

    assert not features[0] < features[0]

    with pytest.raises(TypeError):
        features_sorted[0] < 1


@pytest.mark.parametrize(
    'feature_class, types, expected',
    (
        (Absorption, ['FO', 'SEQ-ZO-FO', 'ZO'], 'ABSORPTION([FO,ZO,SEQ-ZO-FO])'),
        (Elimination, ['FO', 'mm', 'MIX-FO-MM'], 'ELIMINATION([FO,MM,MIX-FO-MM])'),
        (DirectEffect, ['STEP', 'sigmoid', 'LINEAR'], 'DIRECTEFFECT([LINEAR,SIGMOID,STEP])'),
        (EffectComp, ['STEP', 'sigmoid', 'LINEAR'], 'EFFECTCOMP([LINEAR,SIGMOID,STEP])'),
        (Metabolite, ['basic', 'PSC'], 'METABOLITE([PSC,BASIC])'),
    ),
)
def test_repr_many(feature_class, types, expected):
    features = [feature_class.create(type) for type in types]
    mfl1 = ModelFeatures.create(features)
    assert feature_class.repr_many(mfl1) == expected
    mfl2 = ModelFeatures.create([features[0]])
    class_name = feature_class.__name__.upper()
    assert feature_class.repr_many(mfl2) == f'{class_name}({types[0].upper()})'
