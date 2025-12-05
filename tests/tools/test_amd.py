from pathlib import Path

import pytest

from pharmpy.deps import numpy as np
from pharmpy.modeling import (
    create_basic_pk_model,
    has_covariate_effect,
    has_first_order_absorption,
    has_instantaneous_absorption,
    remove_parameter_uncertainty_step,
)
from pharmpy.tools import read_results
from pharmpy.tools.amd.run import (
    _create_model_summary,
    _mechanistic_cov_extraction,
    check_skip,
    create_plots,
    create_start_model,
    create_structural_covariates_model,
    filter_drug_metabolite_dataset,
    filter_tmdd_dataset,
    get_dvid_name,
    get_search_space_covsearch,
    get_search_space_drug_metabolite,
    get_search_space_modelsearch,
    get_search_space_pkpd,
    get_subtool_order,
    later_input_validation,
    modify_search_space_allometry,
    parse_search_space,
    parse_search_space_new,
    split_structural_search_space,
    validate_input,
)
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.run import read_modelfit_results
from pharmpy.workflows import ModelEntry, ModelfitResults
from pharmpy.workflows.contexts import NullContext


def test_create_model_summary(testdata):
    sum_m = read_results(testdata / 'results' / 'modelsearch_results.json').summary_models
    sum_i = read_results(testdata / 'results' / 'iivsearch_results.json').summary_models

    summary_models = _create_model_summary({'modelsearch': sum_m, 'iivsearch': sum_i})

    assert len(summary_models) == len(sum_m) + len(sum_i)


@pytest.mark.parametrize(
    'search_space, error',
    [
        (
            'XYZ',
            'Invalid `search_space`, could not be parsed:',
        ),
        (
            'ELIMINATION(ZO)',  # ABSORPTION(INST) automatically added
            'The given search space have instantaneous absorption',
        ),
    ],
)
def test_invalid_search_space_raises(load_model_for_test, testdata, search_space, error):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError, match=error):
        validate_input(
            model,
            results=res,
            search_space=search_space,
            retries_strategy="skip",
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
        )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_most(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'models' / 'mox2.mod')

    # If no
    model = model.replace(datainfo=model.datainfo.set_types('unknown'))

    validate_input(
        model,
        results=res,
        modeltype='basic_pk',
        administration='oral',
        occasion=None,
        retries_strategy="skip",
        cl_init=1.0,
        vc_init=10.0,
        mat_init=1.0,
    )

    order = get_subtool_order('default')

    to_be_skipped = check_skip(
        NullContext(),
        model,
        modeltype='basic_pk',
        occasion=None,
        order=order,
        allometric_variable=None,
        ignore_datainfo_fallback=False,
        search_space=None,
    )

    assert len(to_be_skipped) == 2

    to_be_skipped = check_skip(
        NullContext(),
        model,
        modeltype='basic_pk',
        occasion=None,
        order=order,
        allometric_variable=None,
        ignore_datainfo_fallback=True,
        search_space=None,
    )

    assert len(to_be_skipped) == 3


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_raise_allometry(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError, match='Invalid `allometric_variable`'):
        later_input_validation(
            model,
            search_space=None,
            allometric_variable='SJDLKSDJ',
            occasion=None,
            mechanistic_covariates=None,
        )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_raise_empty_search_space(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError, match='`search_space` evaluated to be empty :'):
        validate_input(
            model,
            results=res,
            search_space='LET(CONTINUOUS, [AGE, SJDLKSDJ]); LET(CATEGORICAL, [SEX])',
            modeltype='basic_pk',
            administration='oral',
            retries_strategy="skip",
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
        )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_covsearch(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'models' / 'mox2.mod')

    validate_input(
        model,
        results=res,
        modeltype='basic_pk',
        administration='oral',
        occasion='VISI',
        allometric_variable='WT',
        retries_strategy="skip",
        cl_init=1.0,
        vc_init=10.0,
        mat_init=1.0,
        ignore_datainfo_fallback=True,
    )

    order = get_subtool_order('default')

    to_be_skipped = check_skip(
        NullContext(),
        model,
        modeltype='basic_pk',
        occasion='VISI',
        order=order,
        allometric_variable='WT',
        ignore_datainfo_fallback=True,
        search_space=None,
    )

    assert "covariates" in to_be_skipped


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_iovsearch_one_occasion(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'models' / 'mox2.mod')

    validate_input(
        model,
        results=res,
        modeltype='basic_pk',
        administration='oral',
        retries_strategy="skip",
        occasion='XAT2',
        cl_init=1.0,
        vc_init=10.0,
        mat_init=1.0,
    )

    order = get_subtool_order('default')

    to_be_skipped = check_skip(
        NullContext(),
        model,
        modeltype='basic_pk',
        occasion='XAT2',
        allometric_variable=None,
        order=order,
        ignore_datainfo_fallback=False,
        search_space=None,
    )

    assert len(to_be_skipped) == 1


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_iovsearch_missing_occasion_raises(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError, match='Invalid `occasion`'):
        later_input_validation(
            model,
            search_space=None,
            allometric_variable=None,
            occasion='XYZ',
            mechanistic_covariates=None,
        )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_ignore_datainfo_fallback(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'models' / 'mox2.mod')

    validate_input(
        model,
        results=res,
        modeltype='basic_pk',
        administration='oral',
        retries_strategy="skip",
        ignore_datainfo_fallback=True,
        cl_init=1.0,
        vc_init=10.0,
        mat_init=1.0,
    )

    to_be_skipped = check_skip(
        NullContext(),
        model,
        modeltype='basic_pk',
        occasion=None,
        allometric_variable=None,
        order=['modelsearch', 'iivsearch', 'ruvsearch', 'iovsearch', 'allometry', 'covariates'],
        ignore_datainfo_fallback=True,
        search_space=None,
    )

    assert len(to_be_skipped) == 3


@pytest.mark.parametrize(
    'mechanistic_covariates, error',
    [
        (
            ["WT", ("CLCR", "CL")],
            'PASS',
        ),
        (
            ["WT", "CLCR"],
            'PASS',
        ),
        (
            [("CLCR", "CL")],
            'PASS',
        ),
        (
            [("CL", "CLCR")],
            'PASS',
        ),
        (
            ["NOT_A_COVARIATE", ("CLCR", "CL")],
            'Invalid mechanistic covariate:',
        ),
        (
            ["WT", ("CL", "CL")],
            '`mechanistic_covariates` contain invalid argument',
        ),
        (
            ["CLCR", ("WT", "WT")],
            '`mechanistic_covariates` contain invalid argument',
        ),
        (
            ["CLCR", ("WT", "NOT_A_COVARIATE")],
            '`mechanistic_covariates` contain invalid argument',
        ),
        (
            ["CLCR", ("WT",)],
            'Invalid argument in `mechanistic_covariate`:',
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_mechanistic_covariate_option(load_model_for_test, testdata, mechanistic_covariates, error):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    if error != "PASS":
        with pytest.raises(ValueError, match=error):
            later_input_validation(
                model,
                search_space=None,
                allometric_variable=None,
                occasion=None,
                mechanistic_covariates=mechanistic_covariates,
            )
    else:
        # Should not raise any errors
        later_input_validation(
            model,
            search_space=None,
            allometric_variable=None,
            occasion=None,
            mechanistic_covariates=mechanistic_covariates,
        )


@pytest.mark.parametrize(
    'mechanistic_covariates, expected_mechanistic_ss, expected_filtered_ss',
    [
        (["WT", ("CLCR", "CL")], 'COVARIATE?(CL, [WT,CLCR], POW)', ''),
        (["WT", "CLCR"], 'COVARIATE?(CL, [WT,CLCR], POW)', ''),
        ([("CLCR", "CL")], 'COVARIATE?(CL, CLCR, POW)', 'COVARIATE?(CL, WT, POW)'),
        ([("CL", "CLCR")], 'COVARIATE?(CL, CLCR, POW)', 'COVARIATE?(CL, WT, POW)'),
        (
            ["WT"],
            'COVARIATE?(CL, WT, POW)',
            'COVARIATE?(CL, CLCR, POW)',
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_mechanistic_covariate_extraction(
    load_model_for_test,
    testdata,
    mechanistic_covariates,
    expected_mechanistic_ss,
    expected_filtered_ss,
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    search_space = mfl_parse('COVARIATE?(CL, [WT,CLCR], POW)', True)
    mechanistic_ss, filtered_ss = _mechanistic_cov_extraction(
        search_space, model, mechanistic_covariates
    )

    assert mechanistic_ss == mfl_parse(expected_mechanistic_ss, True)
    if expected_filtered_ss:
        assert filtered_ss == mfl_parse(expected_filtered_ss, True)
    else:
        assert not filtered_ss.covariate


@pytest.mark.parametrize(
    'search_space, skipped_expected, parameters_expected',
    [
        ('COVARIATE(CL,WT,exp);COVARIATE?(VC,AGE,exp)', set(), {'CL'}),
        ('COVARIATE(QP1,WT,exp);COVARIATE?(VC,AGE,exp)', {'QP1'}, None),
        ('COVARIATE([CL,VC],WT,exp);COVARIATE?(VC,AGE,exp)', set(), {'CL', 'VC'}),
        ('COVARIATE([CL,QP1],WT,exp);COVARIATE?(VC,AGE,exp)', {'QP1'}, {'CL'}),
    ],
)
def test_split_structural_search_space(
    load_model_for_test, testdata, search_space, skipped_expected, parameters_expected
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    ss_mfl = mfl_parse(search_space, True)
    covsearch_features = ModelFeatures.create(covariate=ss_mfl.covariate)

    skipped, structural = split_structural_search_space(model, covsearch_features)

    assert skipped == skipped_expected
    if parameters_expected is not None:
        assert set(structural.covariate[0].parameter) == parameters_expected
    else:
        assert structural is None


@pytest.mark.parametrize(
    'mfl, expected',
    [
        (
            'ABSORPTION(ZO)',
            'ABSORPTION(ZO);ELIMINATION(FO);TRANSITS(0);PERIPHERALS(0);LAGTIME(OFF)',
        ),
        (None, ModelFeatures()),
        ('', ModelFeatures()),
    ],
)
def test_parse_search_space(mfl, expected):
    if mfl:
        assert repr(parse_search_space(mfl)) == expected
    else:
        assert parse_search_space(mfl) == expected


@pytest.mark.parametrize(
    'mfl, expected_search_space, expected_iiv',
    [
        ('ABSORPTION(ZO);IIV(CL,EXP)', 'ABSORPTION(ZO)', 'IIV(CL,EXP)'),
        (
            'ABSORPTION(ZO);IIV(CL,EXP);COVARIANCE(IIV,[CL,VC])',
            'ABSORPTION(ZO)',
            'IIV(CL,EXP);COVARIANCE(IIV,[CL,VC])',
        ),
        ('IIV(CL,EXP);COVARIANCE(IIV,[CL,VC])', '', 'IIV(CL,EXP);COVARIANCE(IIV,[CL,VC])'),
    ],
)
def test_parse_search_space_new(mfl, expected_search_space, expected_iiv):
    search_space, iiv_features = parse_search_space_new(mfl)
    assert search_space == expected_search_space
    assert repr(iiv_features) == expected_iiv


@pytest.mark.parametrize(
    'mfl, expected_search_space, expected_allometry',
    [
        (
            'ABSORPTION(ZO);ALLOMETRY(WGT,70)',
            'ABSORPTION(ZO);ELIMINATION(FO);TRANSITS(0);PERIPHERALS(0);LAGTIME(OFF)',
            'ALLOMETRY(WGT,70)',
        ),
    ],
)
def test_modify_search_space_allometry(mfl, expected_search_space, expected_allometry):
    ss_mfl = mfl_parse(mfl, mfl_class=True)
    assert ss_mfl.allometry
    ss_mfl, mfl_allometry = modify_search_space_allometry(ss_mfl)
    assert not ss_mfl.allometry
    assert repr(ss_mfl) == expected_search_space
    assert mfl_allometry == mfl_parse(expected_allometry, mfl_class=True).allometry


@pytest.mark.parametrize(
    'modeltype, administration, check_func',
    [
        ('basic_pk', 'iv', has_instantaneous_absorption),
        ('basic_pk', 'oral', has_first_order_absorption),
        ('kpd', 'iv', lambda m: 'POP_KE' in m.parameters.names),
    ],
)
def test_create_start_model(testdata, modeltype, administration, check_func):
    dataset_path = Path(testdata / 'nonmem' / 'pheno.dta')
    cl_init, vc_init, mat_init = 1, 1, 1

    model_from_path = create_start_model(
        dataset_path, modeltype, administration, cl_init, vc_init, mat_init
    )
    assert check_func(model_from_path)

    df = model_from_path.dataset
    model_from_df = create_start_model(df, modeltype, administration, cl_init, vc_init, mat_init)
    assert check_func(model_from_df)

    # Zero protection differs
    assert model_from_path.statements.before_odes == model_from_df.statements.before_odes
    assert model_from_path.statements.ode_system == model_from_df.statements.ode_system


def test_get_dvid_name(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_pd.mod')
    assert get_dvid_name(model) == 'DVID'

    ci = model.datainfo.typeix['dvid'][0]
    ci = ci.replace(type='unknown')
    di = model.datainfo.set_column(ci)
    model = model.replace(datainfo=di)
    assert model.datainfo['DVID'].type == 'unknown'
    assert get_dvid_name(model) == 'DVID'

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert get_dvid_name(model) is None


def test_create_structural_covariates_model(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)
    model_entry = ModelEntry.create(model, modelfit_results=res)

    search_space = 'COVARIATE(CL,WGT,exp);COVARIATE(V,APGR,cat)'
    mfl = mfl_parse(search_space, mfl_class=True)

    model_with_struct = create_structural_covariates_model(mfl, model_entry)

    covariate_effects = [('CL', 'WGT'), ('V', 'APGR')]
    for param, covariate in covariate_effects:
        assert has_covariate_effect(model_with_struct, param, covariate)

    assert model_with_struct.parameters['PTVCL'].init == res.parameter_estimates['PTVCL']


def test_filter_tmdd_dataset(testdata):
    dataset_path = testdata / 'nonmem' / 'pheno.dta'
    model = create_basic_pk_model('iv', dataset_path)
    df = model.dataset
    dvid_col = (np.arange(len(df)) % 3) + 1
    df = model.dataset.assign(DVID=dvid_col)
    model = model.replace(dataset=df)
    assert set(model.dataset['DVID'].values) == {1, 2, 3}
    model_filtered, orig_dataset = filter_tmdd_dataset(model)
    assert set(model.dataset['DVID'].values) == {1, 2, 3}
    assert set(model_filtered.dataset['DVID'].values) == {1}
    assert set(orig_dataset['DVID'].values) == {1, 2, 3}


def test_filter_drug_metabolite_dataset(testdata):
    dataset_path = testdata / 'nonmem' / 'pheno.dta'
    model = create_basic_pk_model('iv', dataset_path)
    df = model.dataset
    dvid_col = (np.arange(len(df)) % 3) + 1
    df = model.dataset.assign(DVID=dvid_col)
    model = model.replace(dataset=df)
    assert set(model.dataset['DVID'].values) == {1, 2, 3}
    model_filtered, orig_dataset = filter_drug_metabolite_dataset(model)
    assert set(model.dataset['DVID'].values) == {1, 2, 3}
    assert set(model_filtered.dataset['DVID'].values) == {1, 3}
    assert set(orig_dataset['DVID'].values) == {1, 2, 3}


@pytest.mark.parametrize(
    'mfl, expected',
    [
        ('DIRECTEFFECT([LINEAR,EMAX]);COVARIATE?(CL,WT,exp)', 'DIRECTEFFECT([LINEAR,EMAX])'),
        ('DIRECTEFFECT([LINEAR,EMAX])', 'DIRECTEFFECT([LINEAR,EMAX])'),
        (
            'COVARIATE?(CL,WT,exp)',
            'DIRECTEFFECT([LINEAR,EMAX,SIGMOID]);'
            'EFFECTCOMP([LINEAR,EMAX,SIGMOID]);'
            'INDIRECTEFFECT([LINEAR,EMAX,SIGMOID],*)',
        ),
        (
            None,
            'DIRECTEFFECT([LINEAR,EMAX,SIGMOID]);'
            'EFFECTCOMP([LINEAR,EMAX,SIGMOID]);'
            'INDIRECTEFFECT([LINEAR,EMAX,SIGMOID],*)',
        ),
    ],
)
def test_get_search_space_pkpd(mfl, expected):
    if mfl is None:
        search_space = ModelFeatures()
    else:
        search_space = mfl_parse(mfl, mfl_class=True)
    search_space_pkpd = get_search_space_pkpd(search_space)
    expected = mfl_parse(expected, mfl_class=True)
    assert search_space_pkpd == expected


@pytest.mark.parametrize(
    'mfl, administration, expected',
    [
        (
            'METABOLITE([PSC,BASIC]);PERIPHERALS(0);PERIPHERALS(1,MET);COVARIATE?(CL,WT,exp)',
            'oral',
            'METABOLITE([PSC,BASIC]);PERIPHERALS(1,MET)',
        ),
        (
            'METABOLITE([PSC,BASIC]);PERIPHERALS(1,MET)',
            'oral',
            'METABOLITE([PSC,BASIC]);PERIPHERALS(1,MET)',
        ),
        (
            'COVARIATE?(CL,WT,exp)',
            'oral',
            'METABOLITE([PSC,BASIC]);PERIPHERALS([0,1],MET)',
        ),
        (
            'COVARIATE?(CL,WT,exp)',
            'ivoral',
            'METABOLITE([PSC,BASIC]);PERIPHERALS([0,1],MET)',
        ),
        (
            'COVARIATE?(CL,WT,exp)',
            'iv',
            'METABOLITE(BASIC);PERIPHERALS([0,1],MET)',
        ),
        (
            None,
            'oral',
            'METABOLITE([PSC,BASIC]);PERIPHERALS([0,1],MET)',
        ),
    ],
)
def test_get_search_space_drug_metabolite(mfl, administration, expected):
    if mfl is None:
        search_space = ModelFeatures()
    else:
        search_space = mfl_parse(mfl, mfl_class=True)
    search_space_met = get_search_space_drug_metabolite(search_space, administration)
    expected = mfl_parse(expected, mfl_class=True)
    assert search_space_met == expected


@pytest.mark.parametrize(
    'mfl, modeltype, administration, expected',
    [
        (
            'ABSORPTION(ZO);COVARIATE?(CL,WT,exp)',
            'basic_pk',
            'iv',
            'ABSORPTION(ZO)',
        ),
        (
            'ABSORPTION(ZO)',
            'basic_pk',
            'iv',
            'ABSORPTION(ZO)',
        ),
        (
            'COVARIATE?(CL,WT,exp)',
            'basic_pk',
            'iv',
            'ELIMINATION(FO);PERIPHERALS(0..2)',
        ),
        (
            None,
            'basic_pk',
            'oral',
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);'
            'ELIMINATION(FO);'
            'LAGTIME([OFF,ON]);'
            'TRANSITS([0,1,3,10],*);'
            'PERIPHERALS(0..1)',
        ),
        (
            None,
            'drug_metabolite',
            'oral',
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);'
            'ELIMINATION(FO);'
            'LAGTIME([OFF,ON]);'
            'TRANSITS([0,1,3,10],*);'
            'PERIPHERALS(0..1)',
        ),
        (
            None,
            'basic_pk',
            'ivoral',
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);'
            'ELIMINATION(FO);'
            'LAGTIME([OFF,ON]);'
            'TRANSITS([0,1,3,10],*);'
            'PERIPHERALS(0..2)',
        ),
        (
            None,
            'drug_metabolite',
            'ivoral',
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);'
            'ELIMINATION(FO);'
            'LAGTIME([OFF,ON]);'
            'TRANSITS([0,1,3,10],*);'
            'PERIPHERALS(0..2)',
        ),
        (
            None,
            'tmdd',
            'oral',
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);'
            'ELIMINATION([MM,MIX-FO-MM]);'
            'LAGTIME([OFF,ON]);'
            'TRANSITS([0,1,3,10],*);'
            'PERIPHERALS(0..1)',
        ),
        (
            None,
            'tmdd',
            'ivoral',
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);'
            'ELIMINATION([MM,MIX-FO-MM]);'
            'LAGTIME([OFF,ON]);'
            'TRANSITS([0,1,3,10],*);'
            'PERIPHERALS(0..2)',
        ),
    ],
)
def test_get_search_space_modelsearch(mfl, modeltype, administration, expected):
    if mfl is None:
        search_space = ModelFeatures()
    else:
        search_space = mfl_parse(mfl, mfl_class=True)
    search_space_modelsearch = get_search_space_modelsearch(search_space, modeltype, administration)
    expected = mfl_parse(expected, mfl_class=True)
    assert search_space_modelsearch == expected


@pytest.mark.parametrize(
    'mfl, modeltype, administration, expected',
    [
        (
            'ABSORPTION(ZO);COVARIATE?(CL,WT,exp)',
            'basic_pk',
            'iv',
            'COVARIATE?(CL,WT,exp)',
        ),
        (
            'COVARIATE?(CL,WT,exp)',
            'basic_pk',
            'iv',
            'COVARIATE?(CL,WT,exp)',
        ),
        (
            'ABSORPTION(ZO)',
            'basic_pk',
            'iv',
            'COVARIATE?(@IIV,@CONTINUOUS,EXP);' 'COVARIATE?(@IIV,@CATEGORICAL,CAT)',
        ),
        (
            None,
            'basic_pk',
            'iv',
            'COVARIATE?(@IIV,@CONTINUOUS,EXP);' 'COVARIATE?(@IIV,@CATEGORICAL,CAT)',
        ),
        (
            None,
            'pkpd',
            'iv',
            'COVARIATE?(@PD_IIV,@CONTINUOUS,EXP);' 'COVARIATE?(@PD_IIV,@CATEGORICAL,CAT)',
        ),
        (
            None,
            'basic_pk',
            'ivoral',
            'COVARIATE?(@IIV,@CONTINUOUS,EXP);'
            'COVARIATE?(@IIV,@CATEGORICAL,CAT);'
            'COVARIATE?(RUV,ADMID,CAT)',
        ),
    ],
)
def test_get_search_space_covsearch(mfl, modeltype, administration, expected):
    if mfl is None:
        search_space = ModelFeatures()
    else:
        search_space = mfl_parse(mfl, mfl_class=True)
    search_space_covsearch = get_search_space_covsearch(search_space, modeltype, administration)
    expected = mfl_parse(expected, mfl_class=True)
    assert str(search_space_covsearch) == str(expected)


@pytest.mark.parametrize(
    'results_to_remove, expected_empty_plots',
    [
        (
            None,
            [],
        ),
        (
            'predictions',
            [
                'final_model_dv_vs_ipred_plot',
                'final_model_dv_vs_pred_plot',
                'final_model_abs_cwres_vs_ipred_plot',
            ],
        ),
        ('residuals', ['final_model_cwres_vs_idv_plot', 'final_model_abs_cwres_vs_ipred_plot']),
        ('individual_estimates', ['final_model_eta_distribution_plot']),
    ],
)
def test_create_plots(load_model_for_test, pheno_path, results_to_remove, expected_empty_plots):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)
    if results_to_remove:
        res_dict = res.to_dict()
        res_dict[results_to_remove] = None
        res = ModelfitResults.from_dict(res_dict)
    plots = create_plots(model, res, model)
    for plot_name, plot in plots.items():
        if plot_name in expected_empty_plots:
            assert not plot
        else:
            assert plot


@pytest.mark.parametrize(
    'kwargs',
    [
        {'cl_init': 1.0, 'vc_init': 1.0, 'mat_init': 1.0},
        {'modeltype': 'pkpd', 'b_init': 1.0, 'emax_init': 1.0, 'ec50_init': 1.0, 'met_init': 1.0},
        {'cl_init': 1.0, 'vc_init': 1.0, 'mat_init': 1.0, 'search_space': 'ABSORPTION(FO)'},
        {
            'cl_init': 1.0,
            'vc_init': 1.0,
            'mat_init': 1.0,
            'search_space': 'ABSORPTION(FO);IIV(CL,exp)',
        },
        {'cl_init': 1.0, 'vc_init': 1.0, 'mat_init': 1.0, 'strictness': 'rse < 0.5'},
        {'cl_init': 1.0, 'vc_init': 1.0, 'mat_init': 1.0, '_E': {'E_p': 0.5}},
    ],
)
def test_validate_input(load_example_model_for_test, kwargs):
    model = load_example_model_for_test('pheno')
    validate_input(model, **kwargs)


@pytest.mark.parametrize(
    'kwargs, exception, match',
    [
        (
            {'modeltype': 'x'},
            ValueError,
            'Invalid `modeltype`',
        ),
        (
            {'administration': 'x'},
            ValueError,
            'Invalid `administration`',
        ),
        (
            {'strategy': 'x'},
            ValueError,
            'Invalid `strategy`',
        ),
        (
            {'retries_strategy': 'x'},
            ValueError,
            'Invalid `retries_strategy`',
        ),
        (
            {'modeltype': 'pkpd', 'cl_init': 1.0},
            ValueError,
            'Cannot provide pk parameter',
        ),
        (
            {
                'modeltype': 'pkpd',
                'b_init': None,
                'emax_init': 1.0,
                'ec50_init': 1.0,
                'met_init': 1.0,
            },
            ValueError,
            'Initial estimate for baseline',
        ),
        (
            {
                'modeltype': 'pkpd',
                'b_init': 1.0,
                'emax_init': None,
                'ec50_init': 1.0,
                'met_init': 1.0,
            },
            ValueError,
            'Initial estimate for E_max',
        ),
        (
            {
                'modeltype': 'pkpd',
                'b_init': 1.0,
                'emax_init': 1.0,
                'ec50_init': None,
                'met_init': 1.0,
            },
            ValueError,
            'Initial estimate for EC_50',
        ),
        (
            {
                'modeltype': 'pkpd',
                'b_init': 1.0,
                'emax_init': 1.0,
                'ec50_init': 1.0,
                'met_init': None,
            },
            ValueError,
            'Initial estimate for MET',
        ),
        (
            {'cl_init': None, 'vc_init': 1.0},
            ValueError,
            'Initial estimate for CL',
        ),
        (
            {'cl_init': 1.0, 'vc_init': None},
            ValueError,
            'Initial estimate for VC',
        ),
        (
            {'administration': 'oral', 'cl_init': 1.0, 'vc_init': 1.0, 'mat_init': None},
            ValueError,
            'Initial estimate for MAT',
        ),
        (
            {
                'cl_init': 1.0,
                'vc_init': 1.0,
                'mat_init': 1.0,
                'parameter_uncertainty_method': None,
                'strictness': 'rse < 0.5',
            },
            ValueError,
            'Invalid `input` model',
        ),
        (
            {'cl_init': 1.0, 'vc_init': 1.0, 'mat_init': 1.0, '_E': {'E_p': 0.0}},
            ValueError,
            'E-values in `_E`',
        ),
        (
            {
                'cl_init': 1.0,
                'vc_init': 1.0,
                'mat_init': 1.0,
                'search_space': 'ABSORPTION(FO);ALLOMETRY(WGT,70)',
                'allometric_variable': 'WGT',
            },
            ValueError,
            'Having both allometric_variable and ALLOMETRY',
        ),
    ],
)
def test_validate_input_raises(load_example_model_for_test, kwargs, exception, match):
    model = load_example_model_for_test('pheno')
    model = remove_parameter_uncertainty_step(model)
    with pytest.raises(exception, match=match):
        validate_input(model, **kwargs)


def test_validate_input_raises_rse_dataset(testdata):
    dataset_path = testdata / 'nonmem' / 'pheno.dta'
    with pytest.raises(ValueError, match='`parameter_uncertainty_method` not set'):
        validate_input(
            dataset_path,
            cl_init=1.0,
            vc_init=1.0,
            mat_init=1.0,
            parameter_uncertainty_method=None,
            strictness='rse < 0.5',
        )


@pytest.mark.parametrize(
    'mfl',
    [
        'COVARIATE(CL,WGT,exp)',
        'COVARIATE(@IIV,WGT,exp)',
        'COVARIATE(@ABSORPTION,WGT,exp)',
    ],
)
def test_later_input_validation_search_space(load_example_model_for_test, mfl):
    model = load_example_model_for_test('pheno')
    later_input_validation(
        model,
        search_space=mfl,
        allometric_variable=None,
        occasion=None,
        mechanistic_covariates=None,
    )


@pytest.mark.parametrize(
    'mfl, exception, match',
    [
        ('COVARIATE(@IIV,WT,exp)', ValueError, 'Invalid `search_space`'),
    ],
)
def test_later_input_validation_search_space_raises(
    load_example_model_for_test, mfl, exception, match
):
    model = load_example_model_for_test('pheno')
    with pytest.raises(exception, match=match):
        later_input_validation(
            model,
            search_space=mfl,
            allometric_variable=None,
            occasion=None,
            mechanistic_covariates=None,
        )


def test_later_input_validation_empty(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    later_input_validation(
        model,
        search_space=None,
        allometric_variable=None,
        occasion=None,
        mechanistic_covariates=None,
    )
