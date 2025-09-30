import pytest

from pharmpy.mfl.features import (
    Absorption,
    Allometry,
    Covariate,
    DirectEffect,
    EffectComp,
    Elimination,
    IndirectEffect,
    LagTime,
    Metabolite,
    ModelFeature,
    Peripherals,
    Transits,
)
from pharmpy.mfl.model_features import ModelFeatures


class ModelFeatureX(ModelFeature):
    @property
    def args(self):
        return tuple()

    def expand(self, model):
        return self

    def repr_many(self):
        return ''


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

    l1 = LagTime.create(on=False)
    l2 = LagTime.create(on=True)
    mf6 = ModelFeatures.create([a1, l1, p1, l2, a2])
    assert mf6.features == (a1, a2, l1, l2, p1)

    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='exp')
    mf7 = ModelFeatures.create([a1, l1, p1, c1, a2])
    assert mf7.features == (a1, a2, l1, p1, c1)

    e1 = Elimination.create(type='MM')
    mf8 = ModelFeatures.create([e1, a2, a1])
    assert mf8.features == (a1, a2, e1)

    de1 = DirectEffect.create(type='LINEAR')
    mf9 = ModelFeatures.create([de1, a2, a1])
    assert mf9.features == (a1, a2, de1)

    ie1 = IndirectEffect.create(type='LINEAR', production_type='PRODUCTION')
    mf10 = ModelFeatures.create([ie1, de1, a2, a1])
    assert mf10.features == (a1, a2, de1, ie1)

    ec1 = EffectComp.create(type='LINEAR')
    mf11 = ModelFeatures.create([ec1, a2, a1])
    assert mf11.features == (a1, a2, ec1)

    m1 = Metabolite.create(type='PSC')
    mf12 = ModelFeatures.create([m1, a2, a1])
    assert mf12.features == (a1, a2, m1)

    al1 = Allometry.create(covariate='WT')
    mf13 = ModelFeatures.create([al1, c1, a2, a1])
    assert mf13.features == (a1, a2, c1, al1)

    with pytest.raises(TypeError):
        ModelFeatures.create(features=1)

    with pytest.raises(TypeError):
        ModelFeatures.create(features=[1])

    with pytest.raises(TypeError):
        ModelFeatures.create(features=[a1, 1])

    x = ModelFeatureX()

    with pytest.raises(NotImplementedError):
        ModelFeatures.create(features=[x])

    al2 = Allometry.create('WT', 80)
    with pytest.raises(ValueError):
        ModelFeatures.create(features=[al1, al2])


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


def test_lagtime():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    l1 = LagTime.create(on=False)
    l2 = LagTime.create(on=True)
    mf = ModelFeatures.create([a1, l2, a2, l1])
    assert mf.features == (a1, a2, l1, l2)
    assert mf.lagtime.features == (l1, l2)


def test_elimination():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    e1 = Elimination.create('FO')
    e2 = Elimination.create('MM')
    mf = ModelFeatures.create([a1, e2, a2, e1])
    assert mf.features == (a1, a2, e1, e2)
    assert mf.elimination.features == (e1, e2)


def test_peripherals():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    p1 = Peripherals.create(0)
    p2 = Peripherals.create(0, 'MET')
    mf = ModelFeatures.create([a1, p2, a2, p1])
    assert mf.features == (a1, a2, p1, p2)
    assert mf.peripherals.features == (p1, p2)


def test_covariates():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    c1 = Covariate.create(parameter='CL', covariate='WGT', fp='EXP')
    c2 = Covariate.create(parameter='VC', covariate='WGT', fp='EXP')
    mf = ModelFeatures.create([a1, c2, a2, c1])
    assert mf.features == (a1, a2, c1, c2)
    assert mf.covariates.features == (c1, c2)


def test_allometry():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    al = Allometry.create('WT')
    mf = ModelFeatures.create([a1, al, a2])
    assert mf.features == (a1, a2, al)
    assert mf.allometry.features == (al,)


def test_direct_effect():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    de1 = DirectEffect.create('LINEAR')
    de2 = DirectEffect.create('SIGMOID')
    mf = ModelFeatures.create([a1, de2, a2, de1])
    assert mf.features == (a1, a2, de1, de2)
    assert mf.direct_effect.features == (de1, de2)


def test_indirect_effect():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    ie1 = IndirectEffect.create('LINEAR', 'DEGRADATION')
    ie2 = IndirectEffect.create('LINEAR', 'PRODUCTION')
    mf = ModelFeatures.create([a1, ie2, a2, ie1])
    assert mf.features == (a1, a2, ie1, ie2)
    assert mf.indirect_effect.features == (ie1, ie2)


def test_effect_comp():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    ec1 = EffectComp.create('LINEAR')
    ec2 = EffectComp.create('SIGMOID')
    mf = ModelFeatures.create([a1, ec2, a2, ec1])
    assert mf.features == (a1, a2, ec1, ec2)
    assert mf.effect_comp.features == (ec1, ec2)


def test_metabolite():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    m1 = Metabolite.create('PSC')
    m2 = Metabolite.create('BASIC')
    mf = ModelFeatures.create([a1, m2, a2, m1])
    assert mf.features == (a1, a2, m1, m2)
    assert mf.metabolite.features == (m1, m2)


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
        (
            [
                Absorption.create('FO'),
                Absorption.create('ZO'),
                LagTime.create(on=False),
                LagTime.create(on=True),
            ],
            'ABSORPTION([FO,ZO]);LAGTIME([OFF,ON])',
        ),
        (
            [
                Absorption.create('FO'),
                Absorption.create('ZO'),
                Allometry.create('WT'),
                Covariate.create(parameter='CL', covariate='WGT', fp='EXP'),
                Covariate.create(parameter='VC', covariate='WGT', fp='EXP'),
            ],
            'ABSORPTION([FO,ZO]);COVARIATE([CL,VC],WGT,EXP,*);ALLOMETRY(WT,70)',
        ),
        (
            [
                Elimination.create('MM'),
                Elimination.create('FO'),
                Absorption.create('FO'),
                Absorption.create('ZO'),
            ],
            'ABSORPTION([FO,ZO]);ELIMINATION([FO,MM])',
        ),
        (
            [
                DirectEffect.create('LINEAR'),
                DirectEffect.create('SIGMOID'),
                EffectComp.create('STEP'),
                EffectComp.create('EMAX'),
                IndirectEffect.create('LINEAR', 'PRODUCTION'),
                IndirectEffect.create('LINEAR', 'DEGRADATION'),
            ],
            'DIRECTEFFECT([LINEAR,SIGMOID]);INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION]);EFFECTCOMP([EMAX,STEP])',
        ),
        (
            [
                Absorption.create('FO'),
                Absorption.create('ZO'),
                Metabolite.create('PSC'),
            ],
            'ABSORPTION([FO,ZO]);METABOLITE(PSC)',
        ),
    ),
)
def test_repr(features, expected):
    mf = ModelFeatures.create(features)
    assert repr(mf) == expected
