from functools import partial

import pytest

from pharmpy.mfl import ModelFeatures
from pharmpy.modeling import (
    add_bioavailability,
    add_covariate_effect,
    add_iiv,
    add_metabolite,
    add_peripheral_compartment,
    set_direct_effect,
    set_instantaneous_absorption,
    set_michaelis_menten_elimination,
    set_mixed_mm_fo_elimination,
    set_seq_zo_fo_absorption,
    set_transit_compartments,
    set_weibull_absorption,
    set_zero_order_absorption,
    set_zero_order_elimination,
)
from pharmpy.modeling.mfl import expand_model_features, generate_transformations, get_model_features


@pytest.mark.parametrize(
    'funcs, source, expected',
    (
        ([], 'COVARIATE?([CL,MAT,VC],[WT,AGE],EXP)', 'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)'),
        ([], 'COVARIATE?(@IIV,[WT,AGE],EXP)', 'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)'),
        ([], 'COVARIATE?(@PK_IIV,[WT,AGE],EXP)', 'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)'),
        (
            [],
            'COVARIATE?([CL,MAT,VC],@CONTINUOUS,EXP)',
            'COVARIATE?([CL,MAT,VC],[AGE,CLCR,WT],EXP,*)',
        ),
        ([], 'COVARIATE?([CL,MAT,VC],@CATEGORICAL,CAT)', 'COVARIATE?([CL,MAT,VC],SEX,CAT,*)'),
        (
            [],
            'LET(CONTINUOUS,[AGE,WT]);COVARIATE?(@IIV,@CONTINUOUS,EXP)',
            'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)',
        ),
        (
            [],
            'COVARIATE([CL,MAT,VC],@CONTINUOUS,EXP)\n'
            'COVARIATE([CL,MAT,VC],@CATEGORICAL,CAT2,+)\n'
            'COVARIATE([CL,MAT,VC],@CATEGORICAL,CAT,+)',
            'COVARIATE([CL,MAT,VC],[AGE,CLCR,WT],EXP,*);COVARIATE([CL,MAT,VC],SEX,[CAT,CAT2],+)',
        ),
        (
            [],
            'COVARIATE?(@PK,@CONTINUOUS,EXP);COVARIATE?(@PK,@CATEGORICAL,CAT)',
            'COVARIATE?([CL,MAT,VC],[AGE,CLCR,WT],EXP,*);COVARIATE?([CL,MAT,VC],SEX,CAT,*)',
        ),
        (
            [],
            'COVARIATE(@ABSORPTION,WT,EXP);COVARIATE(@DISTRIBUTION,AGE,EXP);COVARIATE(@ELIMINATION,SEX,CAT)',
            'COVARIATE(CL,SEX,CAT,*);COVARIATE(MAT,WT,EXP,*);COVARIATE(VC,AGE,EXP,*)',
        ),
        (
            [],
            'COVARIATE(@BIOAVAIL,WT,EXP)',
            '',
        ),
        (
            [add_bioavailability],
            'COVARIATE(@BIOAVAIL,WT,EXP)',
            'COVARIATE(F_BIO,WT,EXP,*)',
        ),
        (
            [],
            'COVARIATE(@PD,WT,EXP)',
            '',
        ),
        (
            [partial(set_direct_effect, expr='linear')],
            'COVARIATE(@PD,WT,EXP)',
            'COVARIATE([B,SLOPE],WT,EXP,*)',
        ),
        (
            [],
            'COVARIATE(@PD_IIV,WT,EXP)',
            '',
        ),
        (
            [partial(set_direct_effect, expr='linear')],
            'COVARIATE(@PD_IIV,WT,EXP)',
            '',
        ),
        (
            [
                partial(set_direct_effect, expr='linear'),
                partial(add_iiv, list_of_parameters='B', expression='exp'),
            ],
            'COVARIATE(@PD_IIV,WT,EXP)',
            'COVARIATE(B,WT,EXP,*)',
        ),
    ),
)
def test_expand_model_features(load_model_for_test, testdata, funcs, source, expected):
    mf = ModelFeatures.create(source)
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)
    mf_expanded = expand_model_features(model, mf)
    assert repr(mf_expanded) == expected


def test_expand_model_features_raises(load_model_for_test, testdata):
    mf = ModelFeatures.create('COVARIATE?(@X,[WGT,AGE],EXP)')
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    with pytest.raises(ValueError):
        expand_model_features(model, mf)


@pytest.mark.parametrize(
    'funcs, type, expected',
    (
        ([], None, 'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)'),
        (
            [set_zero_order_absorption],
            None,
            'ABSORPTION(ZO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        (
            [set_seq_zo_fo_absorption],
            None,
            'ABSORPTION(SEQ-ZO-FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        (
            [set_weibull_absorption],
            None,
            'ABSORPTION(WEIBULL);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        ([set_instantaneous_absorption], None, 'ELIMINATION(FO);PERIPHERALS(0)'),
        (
            [set_mixed_mm_fo_elimination],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(MIX-FO-MM);PERIPHERALS(0)',
        ),
        (
            [set_zero_order_elimination],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(ZO);PERIPHERALS(0)',
        ),
        (
            [set_michaelis_menten_elimination],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(MM);PERIPHERALS(0)',
        ),
        (
            [partial(set_transit_compartments, n=2)],
            None,
            'ABSORPTION(FO);TRANSITS(2);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        (
            [partial(set_transit_compartments, n=2, keep_depot=False)],
            None,
            'ABSORPTION(FO);TRANSITS(2,NODEPOT);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        (
            [add_peripheral_compartment],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(1)',
        ),
        (
            [partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp')],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);COVARIATE(CL,WT,EXP,*)',
        ),
        (
            [
                partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp'),
                partial(add_covariate_effect, parameter='VC', covariate='WT', effect='exp'),
            ],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);COVARIATE([CL,VC],WT,EXP,*)',
        ),
        (
            [
                partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp'),
                partial(add_covariate_effect, parameter='VC', covariate='WT', effect='exp'),
            ],
            'pk',
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        (
            [
                partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp'),
                partial(add_covariate_effect, parameter='VC', covariate='WT', effect='exp'),
            ],
            'covariates',
            'COVARIATE([CL,VC],WT,EXP,*)',
        ),
    ),
)
def test_get_model_features(load_model_for_test, testdata, funcs, type, expected):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)
    mf = get_model_features(model, type)
    assert repr(mf) == expected
    assert mf.is_single_model()


def test_get_model_features_raises(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError):
        get_model_features(model, type='x')


@pytest.mark.parametrize(
    'source, expected',
    (
        ('ABSORPTION(FO)', 1),
        ('ABSORPTION([FO,ZO])', 2),
        ('ABSORPTION(*)', 4),
        ('PERIPHERALS(0..2)', 3),
        ('TRANSITS(N)', 1),
        ('TRANSITS([0,1,3,10],*)', 7),
        ('TRANSITS([0,1,3,10],*);TRANSITS(N)', 8),
        ('LAGTIME([OFF,ON])', 2),
        ('ELIMINATION(*)', 4),
        ('DIRECTEFFECT([LINEAR,SIGMOID]);INDIRECTEFFECT([LINEAR,SIGMOID],*);EFFECTCOMP(EMAX)', 7),
        ('METABOLITE(*)', 2),
        ('ALLOMETRY(WT)', 1),
        ('COVARIATE(CL,WT,EXP)', 1),
        ('COVARIATE([CL,VC,MAT],[WT,AGE],EXP)', 6),
        ('COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)', 12),
        (ModelFeatures.pk_iv(), 6),
        (ModelFeatures.pk_oral(), 15),
    ),
)
def test_generate_transformations(load_model_for_test, testdata, source, expected):
    if isinstance(source, str):
        mf = ModelFeatures.create(source)
    else:
        mf = source
    transformations = generate_transformations(mf)
    assert len(transformations) == expected
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in transformations:
        func(model_start)


def test_generate_transformations_metabolite(load_model_for_test, testdata):
    mf = ModelFeatures.create('PERIPHERALS(0..2);PERIPHERALS(0..2,MET)')
    transformations = generate_transformations(mf)
    assert len(transformations) == 6
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_start = add_metabolite(model_start)
    for func in transformations:
        func(model_start)
