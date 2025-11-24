import pytest

from pharmpy.mfl.features import (
    IIV,
    IOV,
    Absorption,
    Allometry,
    Covariance,
    Covariate,
    DirectEffect,
    EffectComp,
    Elimination,
    IndirectEffect,
    LagTime,
    Metabolite,
    ModelFeature,
    Peripherals,
    Ref,
    Transits,
)
from pharmpy.mfl.model_features import ModelFeatures


class ModelFeatureX(ModelFeature):
    def replace(self, **kwargs):
        return self

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


@pytest.mark.parametrize(
    'features, expected',
    (
        (
            [Absorption.create('FO')],
            [Absorption.create('FO')],
        ),
        (
            [Absorption.create('ZO'), Absorption.create('FO')],
            [Absorption.create('FO'), Absorption.create('ZO')],
        ),
        (
            [Absorption.create('ZO'), Absorption.create('ZO'), Absorption.create('FO')],
            [Absorption.create('FO'), Absorption.create('ZO')],
        ),
        (
            [Peripherals.create(1), Absorption.create('FO'), Peripherals.create(0)],
            [Absorption.create('FO'), Peripherals.create(0), Peripherals.create(1)],
        ),
        (
            [Peripherals.create(1), Absorption.create('FO'), Peripherals.create(0)],
            [Absorption.create('FO'), Peripherals.create(0), Peripherals.create(1)],
        ),
        (
            [
                Transits.create(1),
                Transits.create(0),
                Peripherals.create(1),
                Absorption.create('FO'),
            ],
            [
                Absorption.create('FO'),
                Transits.create(0),
                Transits.create(1),
                Peripherals.create(1),
            ],
        ),
        (
            [
                LagTime.create(True),
                LagTime.create(False),
                Absorption.create('FO'),
                Transits.create(0),
            ],
            [
                Absorption.create('FO'),
                Transits.create(0),
                LagTime.create(False),
                LagTime.create(True),
            ],
        ),
        (
            [
                Covariate.create(parameter='CL', covariate='WGT', fp='exp'),
                Absorption.create('FO'),
                Peripherals.create(0),
            ],
            [
                Absorption.create('FO'),
                Peripherals.create(0),
                Covariate.create(parameter='CL', covariate='WGT', fp='exp'),
            ],
        ),
        (
            [
                Elimination.create(type='MM'),
                Absorption.create('ZO'),
                Elimination.create(type='FO'),
                Absorption.create('FO'),
            ],
            [
                Absorption.create('FO'),
                Absorption.create('ZO'),
                Elimination.create(type='FO'),
                Elimination.create(type='MM'),
            ],
        ),
        (
            [
                DirectEffect.create(type='LINEAR'),
                IndirectEffect.create(type='LINEAR', production=True),
                DirectEffect.create(type='EMAX'),
            ],
            [
                DirectEffect.create(type='LINEAR'),
                DirectEffect.create(type='EMAX'),
                IndirectEffect.create(type='LINEAR', production=True),
            ],
        ),
        (
            [
                EffectComp.create(type='LINEAR'),
                DirectEffect.create(type='LINEAR'),
                Absorption.create('FO'),
            ],
            [
                Absorption.create('FO'),
                DirectEffect.create(type='LINEAR'),
                EffectComp.create(type='LINEAR'),
            ],
        ),
        (
            [
                Metabolite.create(type='BASIC'),
                DirectEffect.create(type='LINEAR'),
                Metabolite.create(type='PSC'),
            ],
            [
                DirectEffect.create(type='LINEAR'),
                Metabolite.create(type='PSC'),
                Metabolite.create(type='BASIC'),
            ],
        ),
        (
            [
                Covariate.create(parameter='CL', covariate='WGT', fp='exp'),
                Absorption.create('FO'),
                Allometry.create(covariate='WGT'),
            ],
            [
                Absorption.create('FO'),
                Covariate.create(parameter='CL', covariate='WGT', fp='exp'),
                Allometry.create(covariate='WGT'),
            ],
        ),
        (
            [
                IOV.create(parameter='CL', fp='exp'),
                IIV.create(parameter='CL', fp='exp'),
                Absorption.create('FO'),
                Peripherals.create(0),
            ],
            [
                Absorption.create('FO'),
                Peripherals.create(0),
                IIV.create(parameter='CL', fp='exp'),
                IOV.create(parameter='CL', fp='exp'),
            ],
        ),
        (
            [
                IOV.create(parameter='CL', fp='exp'),
                IIV.create(parameter='CL', fp='exp'),
                Covariance.create(type='IIV', parameters=['CL', 'VC']),
                Absorption.create('FO'),
            ],
            [
                Absorption.create('FO'),
                IIV.create(parameter='CL', fp='exp'),
                IOV.create(parameter='CL', fp='exp'),
                Covariance.create(type='IIV', parameters=['CL', 'VC']),
            ],
        ),
        (
            [
                IIV.create(parameter='VC', fp='exp', optional=True),
                IIV.create(parameter='CL', fp='exp', optional=True),
                IIV.create(parameter='CL', fp='exp'),
                IIV.create(parameter='MAT', fp='exp', optional=True),
            ],
            [
                IIV.create(parameter='CL', fp='exp'),
                IIV.create(parameter='MAT', fp='exp', optional=True),
                IIV.create(parameter='VC', fp='exp', optional=True),
            ],
        ),
        (
            [
                IIV.create(parameter='CL', fp='exp'),
                IIV.create(parameter='VC', fp='exp', optional=True),
                IIV.create(parameter='CL', fp='exp', optional=True),
                IIV.create(parameter='MAT', fp='exp', optional=True),
            ],
            [
                IIV.create(parameter='CL', fp='exp'),
                IIV.create(parameter='MAT', fp='exp', optional=True),
                IIV.create(parameter='VC', fp='exp', optional=True),
            ],
        ),
    ),
)
def test_create(features, expected):
    mf = ModelFeatures.create(features)
    assert mf.features == tuple(expected)
    assert len(mf) == len(expected)


@pytest.mark.parametrize(
    'mfl, expected',
    (
        (
            'ABSORPTION(FO)',
            [Absorption.create('FO')],
        ),
        (
            'ABSORPTION([ZO,FO])',
            [Absorption.create('FO'), Absorption.create('ZO')],
        ),
        (
            'ABSORPTION([ZO,ZO,FO])',
            [Absorption.create('FO'), Absorption.create('ZO')],
        ),
        (
            'PERIPHERALS(0);ABSORPTION(FO);PERIPHERALS(1)',
            [Absorption.create('FO'), Peripherals.create(0), Peripherals.create(1)],
        ),
        (
            'PERIPHERALS(0..1);ABSORPTION(FO)',
            [Absorption.create('FO'), Peripherals.create(0), Peripherals.create(1)],
        ),
        (
            'TRANSITS([1,0]);PERIPHERALS(1);ABSORPTION(FO)',
            [
                Absorption.create('FO'),
                Transits.create(0),
                Transits.create(1),
                Peripherals.create(1),
            ],
        ),
        (
            'LAGTIME([ON,OFF]);ABSORPTION(FO);TRANSITS(0)',
            [
                Absorption.create('FO'),
                Transits.create(0),
                LagTime.create(False),
                LagTime.create(True),
            ],
        ),
        (
            'COVARIATE(CL,WGT,EXP);ABSORPTION(FO);PERIPHERALS(0)',
            [
                Absorption.create('FO'),
                Peripherals.create(0),
                Covariate.create(parameter='CL', covariate='WGT', fp='exp'),
            ],
        ),
        (
            'ELIMINATION([MM,FO]);ABSORPTION([ZO,FO])',
            [
                Absorption.create('FO'),
                Absorption.create('ZO'),
                Elimination.create(type='FO'),
                Elimination.create(type='MM'),
            ],
        ),
        (
            'DIRECTEFFECT([LINEAR,EMAX]);INDIRECTEFFECT(LINEAR,PRODUCTION)',
            [
                DirectEffect.create(type='LINEAR'),
                DirectEffect.create(type='EMAX'),
                IndirectEffect.create(type='LINEAR', production=True),
            ],
        ),
        (
            'EFFECTCOMP(LINEAR);DIRECTEFFECT(LINEAR);ABSORPTION(FO)',
            [
                Absorption.create('FO'),
                DirectEffect.create(type='LINEAR'),
                EffectComp.create(type='LINEAR'),
            ],
        ),
        (
            'METABOLITE(BASIC);DIRECTEFFECT(LINEAR);METABOLITE(PSC)',
            [
                DirectEffect.create(type='LINEAR'),
                Metabolite.create(type='PSC'),
                Metabolite.create(type='BASIC'),
            ],
        ),
        (
            'ABSORPTION(FO);COVARIATE(CL,WGT,EXP);ALLOMETRY(WGT,70)',
            [
                Absorption.create('FO'),
                Covariate.create(parameter='CL', covariate='WGT', fp='exp'),
                Allometry.create(covariate='WGT'),
            ],
        ),
        (
            'IIV(CL,EXP);ABSORPTION(FO);PERIPHERALS(0)',
            [
                Absorption.create('FO'),
                Peripherals.create(0),
                IIV.create(parameter='CL', fp='exp'),
            ],
        ),
        (
            'IIV(CL,EXP);IOV(CL,EXP);PERIPHERALS(0);ABSORPTION(FO)',
            [
                Absorption.create('FO'),
                Peripherals.create(0),
                IIV.create(parameter='CL', fp='exp'),
                IOV.create(parameter='CL', fp='exp'),
            ],
        ),
        (
            'IIV(CL,EXP);IOV(CL,EXP);COVARIANCE(IIV,[CL,VC]);ABSORPTION(FO)',
            [
                Absorption.create('FO'),
                IIV.create(parameter='CL', fp='exp'),
                IOV.create(parameter='CL', fp='exp'),
                Covariance.create(type='IIV', parameters=['CL', 'VC']),
            ],
        ),
    ),
)
def test_create_from_mfl(mfl, expected):
    mf = ModelFeatures.create(mfl)
    assert mf.features == tuple(expected)


def test_create_raises():
    with pytest.raises(TypeError):
        ModelFeatures.create(features=1)

    with pytest.raises(TypeError):
        ModelFeatures.create(features=[1])

    with pytest.raises(ValueError):
        ModelFeatures.create(features='x')

    al1 = Allometry.create('WT')
    with pytest.raises(TypeError):
        ModelFeatures.create(features=[al1, 1])

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


def test_pk_iv():
    mf = ModelFeatures.pk_iv()
    assert len(mf) == 6
    assert repr(mf) == 'TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0..2)'


def test_pk_oral():
    mf = ModelFeatures.pk_oral()
    assert len(mf) == 16
    assert repr(mf) == (
        'ABSORPTION([FO,ZO,SEQ-ZO-FO]);TRANSITS([0,1,3,10],[DEPOT,NODEPOT]);'
        'LAGTIME([OFF,ON]);ELIMINATION(FO);PERIPHERALS(0..1)'
    )


def test_pd():
    mf = ModelFeatures.pd()
    assert len(mf) == 12
    assert repr(mf) == (
        'DIRECTEFFECT([LINEAR,EMAX,SIGMOID]);'
        'INDIRECTEFFECT([LINEAR,EMAX,SIGMOID],[DEGRADATION,PRODUCTION]);'
        'EFFECTCOMP([LINEAR,EMAX,SIGMOID])'
    )


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
    p2 = Peripherals.create(0, True)
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


def test_direct_effecta():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    de1 = DirectEffect.create('LINEAR')
    de2 = DirectEffect.create('SIGMOID')
    mf = ModelFeatures.create([a1, de2, a2, de1])
    assert mf.features == (a1, a2, de1, de2)
    assert mf.direct_effects.features == (de1, de2)


def test_indirect_effects():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    ie1 = IndirectEffect.create('LINEAR', production=False)
    ie2 = IndirectEffect.create('LINEAR', production=True)
    mf = ModelFeatures.create([a1, ie2, a2, ie1])
    assert mf.features == (a1, a2, ie1, ie2)
    assert mf.indirect_effects.features == (ie1, ie2)


def test_effect_compartments():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    ec1 = EffectComp.create('LINEAR')
    ec2 = EffectComp.create('SIGMOID')
    mf = ModelFeatures.create([a1, ec2, a2, ec1])
    assert mf.features == (a1, a2, ec1, ec2)
    assert mf.effect_compartments.features == (ec1, ec2)


def test_metabolites():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    m1 = Metabolite.create('PSC')
    m2 = Metabolite.create('BASIC')
    mf = ModelFeatures.create([a1, m2, a2, m1])
    assert mf.features == (a1, a2, m1, m2)
    assert mf.metabolites.features == (m1, m2)


def test_iiv():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    iiv1 = IIV.create('CL', 'EXP')
    iiv2 = IIV.create('MAT', 'EXP')
    mf = ModelFeatures.create([a1, iiv2, a2, iiv1])
    assert mf.features == (a1, a2, iiv1, iiv2)
    assert mf.iiv.features == (iiv1, iiv2)


def test_iov():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    iov = IOV.create('CL', 'EXP')
    iiv = IIV.create('MAT', 'EXP')
    mf = ModelFeatures.create([a1, iov, a2, iiv])
    assert mf.features == (a1, a2, iiv, iov)
    assert mf.iov.features == (iov,)


def test_covariance():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    cov1 = Covariance.create('IIV', ['CL', 'VC'])
    cov2 = Covariance.create('IOV', ['CL', 'VC'])
    mf = ModelFeatures.create([a1, cov2, a2, cov1])
    assert mf.features == (a1, a2, cov1, cov2)
    assert mf.covariance.features == (cov1, cov2)


def test_refs():
    mf = ModelFeatures.pk_oral()
    assert mf.refs == tuple()
    mf += Covariate.create(Ref('IIV'), 'WGT', 'exp')
    assert mf.refs == (Ref('IIV'),)
    mf += Covariate.create(Ref('ABSORPTION'), 'AGE', 'exp')
    assert mf.refs == (
        Ref('ABSORPTION'),
        Ref('IIV'),
    )


def test_is_expanded():
    features = [Absorption.create('FO'), Absorption.create('SEQ-ZO-FO')]
    mf1 = ModelFeatures.create(features)
    assert mf1.is_expanded()
    features.append(Covariate.create(Ref('IIV'), 'WGT', 'exp'))
    mf2 = ModelFeatures.create(features)
    assert not mf2.is_expanded()


@pytest.mark.parametrize(
    'features, expected',
    (
        (
            [Absorption.create('FO'), Elimination.create('FO')],
            True,
        ),
        (
            [Absorption.create('FO'), Absorption.create('ZO')],
            False,
        ),
        (
            [Peripherals.create(0), Peripherals.create(1, True)],
            True,
        ),
        (
            [Peripherals.create(0), Peripherals.create(1)],
            False,
        ),
        (
            [Peripherals.create(0), Peripherals.create(0, True), Peripherals.create(1, True)],
            False,
        ),
        (
            [Covariate.create('CL', 'WGT', 'exp'), Covariate.create('VC', 'WGT', 'exp')],
            True,
        ),
        (
            [
                Covariate.create('CL', 'WGT', 'exp', optional=True),
                Covariate.create('VC', 'WGT', 'exp'),
            ],
            False,
        ),
        (
            [IIV.create('CL', 'exp', optional=False), IIV.create('VC', 'exp', optional=False)],
            True,
        ),
        (
            [
                IIV.create('CL', 'exp', optional=True),
                IIV.create('VC', 'exp', optional=False),
            ],
            False,
        ),
        (
            [
                IIV.create('CL', 'exp', optional=False),
                IIV.create('CL', 'add', optional=False),
            ],
            False,
        ),
        (
            [IOV.create('CL', 'exp', optional=False), IOV.create('VC', 'exp', optional=False)],
            True,
        ),
        (
            [
                IOV.create('CL', 'exp', optional=True),
                IOV.create('VC', 'exp', optional=False),
            ],
            False,
        ),
        (
            [
                Covariance.create('IIV', ['CL', 'VC'], optional=False),
                Covariance.create('IIV', ['CL', 'MAT'], optional=False),
            ],
            True,
        ),
        (
            [
                Covariance.create('IIV', ['CL', 'VC'], optional=True),
                Covariance.create('IIV', ['CL', 'MAT'], optional=False),
            ],
            False,
        ),
    ),
)
def test_is_single_model(features, expected):
    mf = ModelFeatures.create(features)
    assert mf.is_single_model() == expected


def test_is_single_model_raises():
    mf = ModelFeatures([ModelFeatureX()])
    with pytest.raises(NotImplementedError):
        mf.is_single_model()


def test_expand():
    mf = ModelFeatures.pk_iv()
    expand_to = {Ref('IIV'): ['CL', 'VC', 'MAT']}
    assert mf.is_expanded()
    assert mf.expand(expand_to) == mf
    mf += Covariate.create(parameter=Ref('IIV'), covariate='WGT', fp='EXP')
    assert not mf.is_expanded()
    mf_expanded = mf.expand(expand_to)
    assert len(mf_expanded) == len(mf) + 2
    assert mf_expanded.is_expanded()

    mf = ModelFeatures.create('IIV(CL,EXP);IIV?(@IIV,[ADD,EXP])')
    assert not mf.is_expanded()
    mf_expanded = mf.expand(expand_to)
    assert len(mf_expanded) == len(mf) + 2
    assert mf_expanded.is_expanded()

    a = Absorption(type=Ref('x'))
    mf_incorrect = ModelFeatures(mf.features + (a,))
    with pytest.raises(NotImplementedError):
        mf_incorrect.expand(expand_to)


def test_filter():
    iiv1 = IIV.create(parameter='CL', fp='exp', optional=False)
    iiv2 = IIV.create(parameter='MAT', fp='exp', optional=True)
    iiv3 = IIV.create(parameter='VC', fp='exp', optional=True)

    mf = ModelFeatures.create([iiv1, iiv2, iiv3])

    assert mf.filter(filter_on='optional').features == (iiv2, iiv3)
    assert mf.filter(filter_on='forced').features == (iiv1,)

    mf2 = ModelFeatures.pk_oral()
    mf2_iiv = mf2 + iiv1
    assert mf2_iiv.filter(filter_on='pk') == mf2

    with pytest.raises(NotImplementedError):
        mf.filter(filter_on='x')


def test_force_optional():
    iiv1 = IIV.create(parameter='CL', fp='exp', optional=False)
    iiv2 = IIV.create(parameter='MAT', fp='exp', optional=False)
    mf1 = ModelFeatures.create([iiv1, iiv2])
    assert mf1 == mf1.force_optional()

    iiv3 = IIV.create(parameter='VC', fp='exp', optional=True)
    mf2 = mf1 + iiv3
    mf2_forced = mf2.force_optional()
    assert mf2_forced != mf2
    assert all(feature.optional is False for feature in mf2_forced)


@pytest.mark.parametrize(
    'features1, features2, expected',
    [
        (
            [Absorption.create('FO'), Elimination.create('FO')],
            [Absorption.create('FO'), Elimination.create('FO'), Elimination.create('ZO')],
            True,
        ),
        (
            [
                Absorption.create('FO'),
                Elimination.create('FO'),
                Covariate.create('CL', 'WGT', 'exp', optional=False),
            ],
            [Absorption.create('FO'), Elimination.create('FO'), Elimination.create('ZO')],
            False,
        ),
        (
            [
                Absorption.create('FO'),
                Elimination.create('FO'),
                Covariate.create('CL', 'WGT', 'exp', optional=True),
            ],
            [Absorption.create('FO'), Elimination.create('FO'), Elimination.create('ZO')],
            False,
        ),
        (
            [Absorption.create('FO'), Elimination.create('FO')],
            [
                Absorption.create('FO'),
                Elimination.create('FO'),
                Elimination.create('ZO'),
                Covariate.create('CL', 'WGT', 'exp', optional=True),
            ],
            True,
        ),
    ],
)
def test_contains(features1, features2, expected):
    mf1 = ModelFeatures.create(features1)
    mf2 = ModelFeatures.create(features2)

    assert (mf1 in mf2) == expected
    assert (features1 in mf2) == expected
    assert all(f in mf2 for f in features1) == expected
    assert 1 not in mf1
    assert [1] not in mf1

    # mf1 = ModelFeatures.pk_oral()
    #
    # a = Absorption.create('FO')
    # e = Elimination.create('FO')
    # mf2 = ModelFeatures.create([a, e])
    #
    # assert mf2 in mf1
    # assert mf1 not in mf2
    # assert a in mf1 and a in mf2
    # assert [a, e] in mf1
    #
    # c = Covariate.create('CL', 'WGT', 'exp')
    # assert [c] not in mf1
    #
    # mf3 = ModelFeatures.create([a, e, c])
    # assert mf3 not in mf1
    #
    # assert 1 not in mf1
    #
    # c_optional = c.replace(optional=True)
    # mf4 = ModelFeatures.create([a, e, c_optional])
    # assert c in mf4
    # assert c_optional not in mf3


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


def test_sub():
    a1 = Absorption.create('FO')
    a2 = Absorption.create('ZO')
    mf1 = ModelFeatures.create([a1, a2])

    mf2 = mf1 - a1
    assert mf1.features == (a1, a2)
    assert mf2.features == (a2,)

    mf3 = mf1 - [a1]
    assert mf3 == mf2

    mf4 = mf1 - ModelFeatures.create([a1])
    assert mf4 == mf2

    a3 = Absorption.create('SEQ-ZO-FO')
    mf5 = mf1 - a3
    assert mf1 == mf5

    mf6 = a1 - mf1
    assert mf6.features == tuple()

    mf7 = [a1] - mf1
    assert mf7 == mf6

    with pytest.raises(TypeError):
        mf1 - 1

    with pytest.raises(TypeError):
        1 - mf1


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
                Peripherals.create(0, True),
                Absorption.create('ZO'),
            ],
            'ABSORPTION([FO,ZO]);PERIPHERALS(0..1);PERIPHERALS(0,MET)',
        ),
        (
            [
                Peripherals.create(0),
                Peripherals.create(1),
                Peripherals.create(0, True),
                Transits.create(0),
                Transits.create(1),
                Transits.create(3),
                Transits.create(0, False),
                Transits.create(1, False),
                Transits.create(3, False),
            ],
            'TRANSITS([0,1,3],[DEPOT,NODEPOT]);PERIPHERALS(0..1);PERIPHERALS(0,MET)',
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
                IndirectEffect.create('LINEAR', production=True),
                IndirectEffect.create('LINEAR', production=False),
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
        (
            [
                IIV.create('VC', 'EXP', True),
                Covariance.create('IIV', ['MAT', 'VC'], True),
                Covariance.create('IIV', ['CL', 'VC'], True),
                IOV.create('CL', 'EXP', True),
                IOV.create('VC', 'EXP', True),
                Covariance.create('IIV', ['MAT', 'CL'], True),
                IIV.create('CL', 'EXP', True),
                IIV.create('MAT', 'EXP', True),
            ],
            'IIV?([CL,MAT,VC],EXP);IOV?([CL,VC],EXP);COVARIANCE?(IIV,[CL,MAT,VC])',
        ),
    ),
)
def test_repr(features, expected):
    mf = ModelFeatures.create(features)
    assert repr(mf) == expected


@pytest.mark.parametrize(
    'source',
    (
        'ABSORPTION([FO,ZO,SEQ-ZO-FO])',
        'ABSORPTION([FO,ZO]);PERIPHERALS(0..2)',
        'ABSORPTION([FO,ZO]);PERIPHERALS(0..1);PERIPHERALS(0,MET)',
        'TRANSITS([0,1,3],[DEPOT,NODEPOT]);PERIPHERALS(0..1);PERIPHERALS(0,MET)',
        'ABSORPTION([FO,ZO]);LAGTIME([OFF,ON])',
        'ABSORPTION([FO,ZO]);COVARIATE([CL,VC],WGT,EXP,*);ALLOMETRY(WT,70)',
        'ABSORPTION([FO,ZO]);ELIMINATION([FO,MM])',
        'DIRECTEFFECT([LINEAR,SIGMOID]);INDIRECTEFFECT(LINEAR,[DEGRADATION,PRODUCTION]);EFFECTCOMP([EMAX,STEP])',
        'ABSORPTION([FO,ZO]);METABOLITE(PSC)',
        'IIV([CL,MAT,VC],EXP);IOV([CL,MAT,VC],EXP)',
        'COVARIANCE(IIV,[CL,MAT,VC])',
    ),
)
def test_parse_repr_roundtrip(source):
    mf = ModelFeatures.create(source)
    assert repr(mf) == source
