from functools import partial

import pytest

from pharmpy.mfl import ModelFeatures
from pharmpy.modeling import (
    add_allometry,
    add_bioavailability,
    add_covariate_effect,
    add_effect_compartment,
    add_iiv,
    add_indirect_effect,
    add_lag_time,
    add_metabolite,
    add_peripheral_compartment,
    create_joint_distribution,
    remove_covariate_effect,
    remove_iiv,
    remove_lag_time,
    set_direct_effect,
    set_first_order_absorption,
    set_first_order_elimination,
    set_instantaneous_absorption,
    set_michaelis_menten_elimination,
    set_mixed_mm_fo_elimination,
    set_n_transit_compartments,
    set_peripheral_compartments,
    set_seq_zo_fo_absorption,
    set_transit_compartments,
    set_weibull_absorption,
    set_zero_order_absorption,
    set_zero_order_elimination,
    split_joint_distribution,
)
from pharmpy.modeling.mfl import (
    expand_model_features,
    generate_transformations,
    get_model_features,
    is_in_search_space,
    transform_into_search_space,
)


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
        (
            [],
            'IIV(@PK,EXP)',
            'IIV([CL,MAT,VC],EXP)',
        ),
        ([], 'COVARIANCE(IIV,@IIV)', 'COVARIANCE(IIV,[CL,MAT,VC])'),
        ([], 'IIV(CL,EXP);IIV?(@PK,[ADD,EXP])', 'IIV(CL,EXP);IIV?([MAT,VC],[ADD,EXP])'),
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
        (
            [],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);IIV([CL,MAT,VC],EXP)',
        ),
        (
            [set_zero_order_absorption],
            None,
            'ABSORPTION(ZO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);IIV([CL,D1,VC],EXP)',
        ),
        (
            [set_seq_zo_fo_absorption],
            None,
            'ABSORPTION(SEQ-ZO-FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);IIV([CL,MAT,VC],EXP)',
        ),
        (
            [set_weibull_absorption],
            None,
            'ABSORPTION(WEIBULL);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);IIV([CL,VC],EXP)',
        ),
        ([set_instantaneous_absorption], None, 'ELIMINATION(FO);PERIPHERALS(0);IIV([CL,VC],EXP)'),
        (
            [set_mixed_mm_fo_elimination],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(MIX-FO-MM);PERIPHERALS(0);IIV([CL,MAT,VC],EXP)',
        ),
        (
            [set_zero_order_elimination],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(ZO);PERIPHERALS(0);IIV([CLMM,MAT,VC],EXP)',
        ),
        (
            [set_michaelis_menten_elimination],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(MM);PERIPHERALS(0);IIV([CLMM,MAT,VC],EXP)',
        ),
        (
            [partial(set_transit_compartments, n=2)],
            None,
            'ABSORPTION(FO);TRANSITS(2);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);IIV([CL,MAT,VC],EXP)',
        ),
        (
            [partial(set_transit_compartments, n=2, keep_depot=False)],
            None,
            'ABSORPTION(FO);TRANSITS(2,NODEPOT);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);IIV([CL,MDT,VC],EXP)',
        ),
        (
            [add_peripheral_compartment],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(1);IIV([CL,MAT,VC],EXP)',
        ),
        (
            [partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp')],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);'
            'COVARIATE(CL,WT,EXP,*);IIV([CL,MAT,VC],EXP)',
        ),
        (
            [
                partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp'),
                partial(add_covariate_effect, parameter='VC', covariate='WT', effect='exp'),
            ],
            None,
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0);'
            'COVARIATE([CL,VC],WT,EXP,*);IIV([CL,MAT,VC],EXP)',
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
        (
            [],
            'iiv',
            'IIV([CL,MAT,VC],EXP)',
        ),
        (
            [
                partial(remove_iiv, to_remove=['CL']),
                partial(add_iiv, list_of_parameters=['CL'], expression='add'),
            ],
            'iiv',
            'IIV(CL,ADD);IIV([MAT,VC],EXP)',
        ),
        (
            [
                partial(add_covariate_effect, parameter='CL', covariate='WT', effect='lin'),
                partial(remove_iiv, to_remove=['CL']),
                partial(add_iiv, list_of_parameters=['CL'], expression='add'),
            ],
            'iiv',
            'IIV(CL,ADD);IIV([MAT,VC],EXP)',
        ),
        (
            [
                partial(add_covariate_effect, parameter='VC', covariate='WT', effect='lin'),
                partial(remove_iiv, to_remove=['CL']),
                partial(add_iiv, list_of_parameters=['CL'], expression='add'),
            ],
            'iiv',
            'IIV(CL,ADD);IIV([MAT,VC],EXP)',
        ),
        (
            [create_joint_distribution],
            'covariance',
            'COVARIANCE(IIV,[CL,MAT,VC])',
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

    model = remove_iiv(model, 'CL')
    model = add_iiv(model, 'CL', 'log')
    with pytest.raises(NotImplementedError):
        get_model_features(model, type='iiv')


@pytest.mark.parametrize(
    'source, expected, allowed_functions',
    (
        ('ABSORPTION(FO)', 1, (set_first_order_absorption,)),
        ('ABSORPTION([FO,ZO])', 2, (set_first_order_absorption, set_zero_order_absorption)),
        (
            'ABSORPTION(*)',
            4,
            (
                set_first_order_absorption,
                set_zero_order_absorption,
                set_seq_zo_fo_absorption,
                set_weibull_absorption,
            ),
        ),
        ('PERIPHERALS(0..2)', 3, (set_peripheral_compartments,)),
        ('TRANSITS(N)', 1, (set_n_transit_compartments,)),
        ('TRANSITS([0,1,3,10],*)', 7, (set_transit_compartments,)),
        (
            'TRANSITS([0,1,3,10],*);TRANSITS(N)',
            8,
            (
                set_transit_compartments,
                set_n_transit_compartments,
            ),
        ),
        (
            'LAGTIME([OFF,ON])',
            2,
            (
                remove_lag_time,
                add_lag_time,
            ),
        ),
        (
            'ELIMINATION(*)',
            4,
            (
                set_first_order_elimination,
                set_zero_order_elimination,
                set_mixed_mm_fo_elimination,
                set_michaelis_menten_elimination,
            ),
        ),
        (
            'DIRECTEFFECT([LINEAR,SIGMOID]);INDIRECTEFFECT([LINEAR,SIGMOID],*);EFFECTCOMP(EMAX)',
            7,
            (set_direct_effect, add_indirect_effect, add_effect_compartment),
        ),
        ('METABOLITE(*)', 2, (add_metabolite,)),
        ('ALLOMETRY(WT)', 1, (add_allometry,)),
        ('COVARIATE(CL,WT,EXP)', 1, (add_covariate_effect,)),
        ('COVARIATE([CL,VC,MAT],[WT,AGE],EXP)', 6, (add_covariate_effect,)),
        (
            'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)',
            12,
            (add_covariate_effect, remove_covariate_effect),
        ),
        ('IIV([CL,MAT,VC],EXP)', 6, (add_iiv, remove_iiv)),
        (
            'IIV?([CL,MAT,VC],EXP)',
            6,
            (
                add_iiv,
                remove_iiv,
            ),
        ),
        (
            'COVARIANCE(IIV,[CL,MAT,VC])',
            4,
            (create_joint_distribution, split_joint_distribution),
        ),
    ),
)
def test_generate_transformations(
    load_model_for_test, testdata, source, expected, allowed_functions
):
    if isinstance(source, str):
        mf = ModelFeatures.create(source)
    else:
        mf = source
    transformations = generate_transformations(mf)
    assert len(transformations) == expected
    assert all(
        f.func in allowed_functions if hasattr(f, 'func') else f in allowed_functions
        for f in transformations
    )


def test_generate_transformations_metabolite(load_model_for_test, testdata):
    mf = ModelFeatures.create('PERIPHERALS(0..2);PERIPHERALS(0..2,MET)')
    transformations = generate_transformations(mf)
    assert len(transformations) == 6
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_start = add_metabolite(model_start)
    for func in transformations:
        func(model_start)


@pytest.mark.parametrize(
    'funcs, source, type, expected',
    (
        (
            [],
            'ABSORPTION(ZO)',
            'pk',
            'ABSORPTION(ZO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        (
            [set_zero_order_absorption],
            'ABSORPTION(FO)',
            'pk',
            'ABSORPTION(FO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(0)',
        ),
        (
            [],
            'ABSORPTION([ZO,SEQ-ZO-FO]);TRANSITS([0,1,3,10],[DEPOT,NODEPOT]);'
            'LAGTIME([OFF,ON]);ELIMINATION(FO);PERIPHERALS(1)',
            'pk',
            'ABSORPTION(ZO);TRANSITS(0);LAGTIME(OFF);ELIMINATION(FO);PERIPHERALS(1)',
        ),
        (
            [],
            'IIV([CL,MAT,VC],EXP)',
            'iiv',
            'IIV([CL,MAT,VC],EXP)',
        ),
        (
            [partial(remove_iiv, to_remove='CL')],
            'IIV?([CL,MAT,VC],EXP)',
            'iiv',
            'IIV([MAT,VC],EXP)',
        ),
        (
            [add_lag_time],
            'IIV([CL,MAT,MDT,VC],[EXP,ADD])',
            'iiv',
            'IIV([CL,MAT,MDT,VC],EXP)',
        ),
        (
            [],
            'IIV([MAT,VC],EXP)',
            'iiv',
            'IIV([MAT,VC],EXP)',
        ),
        (
            [],
            'IIV([MAT,VC],EXP)',
            'iiv',
            'IIV([MAT,VC],EXP)',
        ),
        (
            [],
            'IIV(CL,ADD);IIV([MAT,VC],EXP)',
            'iiv',
            'IIV(CL,ADD);IIV([MAT,VC],EXP)',
        ),
        (
            [],
            'IIV(CL,ADD);IIV([MAT,VC],[ADD,EXP])',
            'iiv',
            'IIV(CL,ADD);IIV([MAT,VC],EXP)',
        ),
        (
            [],
            'COVARIANCE(IIV,[CL,MAT,VC])',
            'covariance',
            'COVARIANCE(IIV,[CL,MAT,VC])',
        ),
        (
            [],
            'IIV(CL,EXP);IIV([MAT,VC],[EXP,ADD])',
            'iiv',
            'IIV([CL,MAT,VC],EXP)',
        ),
        (
            [],
            [],
            'iiv',
            '',
        ),
        (
            [],
            'COVARIANCE(IIV,[CL,MAT])',
            'covariance',
            'COVARIANCE(IIV,[CL,MAT])',
        ),
        (
            [create_joint_distribution],
            'COVARIANCE(IIV,[CL,MAT])',
            'covariance',
            'COVARIANCE(IIV,[CL,MAT])',
        ),
        (
            [
                add_peripheral_compartment,
                partial(add_iiv, list_of_parameters=['QP1', 'VP1'], expression='exp'),
                create_joint_distribution,
            ],
            'COVARIANCE(IIV,[CL,MAT])',
            'covariance',
            'COVARIANCE(IIV,[CL,MAT])',
        ),
        (
            [
                add_peripheral_compartment,
                partial(add_iiv, list_of_parameters=['QP1', 'VP1'], expression='exp'),
                create_joint_distribution,
            ],
            'COVARIANCE(IIV,[CL,MAT,VC])',
            'covariance',
            'COVARIANCE(IIV,[CL,MAT,VC])',
        ),
        (
            [
                add_peripheral_compartment,
                partial(add_iiv, list_of_parameters=['QP1', 'VP1'], expression='exp'),
                create_joint_distribution,
            ],
            [],
            'covariance',
            '',
        ),
    ),
)
def test_transform_into_search_space(load_model_for_test, testdata, funcs, source, type, expected):
    mf = ModelFeatures.create(source)
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)
    model_transformed = transform_into_search_space(model, mf, type=type)
    assert repr(get_model_features(model_transformed, type=type)) == expected


@pytest.mark.parametrize(
    'funcs, source, type, expected',
    (
        ([], 'ABSORPTION(ZO)', 'pk', False),
        ([set_zero_order_absorption], 'ABSORPTION(FO)', 'pk', False),
        (
            [],
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);TRANSITS([0,1,3,10],[DEPOT,NODEPOT]);'
            'LAGTIME([OFF,ON]);ELIMINATION(FO);PERIPHERALS(0..1)',
            'pk',
            True,
        ),
        (
            [],
            'ABSORPTION([ZO,SEQ-ZO-FO]);TRANSITS([0,1,3,10],[DEPOT,NODEPOT]);'
            'LAGTIME([OFF,ON]);ELIMINATION(FO);PERIPHERALS(1)',
            'pk',
            False,
        ),
        (
            [],
            'IIV([CL,MAT,VC],EXP)',
            'iiv',
            True,
        ),
        (
            [partial(remove_iiv, to_remove='CL')],
            'IIV?([CL,MAT,VC],EXP)',
            'iiv',
            True,
        ),
        (
            [add_lag_time],
            'IIV?([CL,MAT,MDT,VC],[EXP,ADD])',
            'iiv',
            True,
        ),
        (
            [add_lag_time],
            'IIV([CL,MAT,MDT,VC],[EXP,ADD])',
            'iiv',
            False,
        ),
        (
            [],
            'IIV(CL,ADD);IIV([MAT,VC],[ADD,EXP])',
            'iiv',
            False,
        ),
        (
            [],
            'COVARIANCE(IIV,[CL,MAT,VC])',
            'covariance',
            False,
        ),
        (
            [],
            'IIV(CL,EXP);IIV([MAT,VC],[EXP,ADD])',
            'iiv',
            True,
        ),
        (
            [],
            [],
            'iiv',
            False,
        ),
        (
            [],
            'COVARIANCE(IIV,[CL,MAT])',
            'covariance',
            False,
        ),
        (
            [create_joint_distribution],
            'COVARIANCE(IIV,[CL,MAT,VC])',
            'covariance',
            True,
        ),
    ),
)
def test_is_in_search_space(load_model_for_test, testdata, funcs, source, type, expected):
    mf = ModelFeatures.create(source)
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)
    assert is_in_search_space(model, mf, type=type) == expected
