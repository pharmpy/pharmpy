import shutil
from pathlib import Path

import pytest

from pharmpy.model import Model
from pharmpy.tools import read_modelfit_results, read_results
from pharmpy.tools.amd.run import (
    _create_model_summary,
    _mechanistic_cov_extraction,
    check_skip,
    later_input_validation,
    validate_input,
)
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.workflows import default_context
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
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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

    to_be_skipped = check_skip(
        NullContext(),
        model,
        occasion=None,
        allometric_variable=None,
        ignore_datainfo_fallback=False,
        search_space=None,
    )

    assert len(to_be_skipped) == 2

    to_be_skipped = check_skip(
        NullContext(),
        model,
        occasion=None,
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
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError, match='Invalid `allometric_variable`'):
        later_input_validation(
            model,
            results=res,
            modeltype='basic_pk',
            administration='oral',
            allometric_variable='SJDLKSDJ',
            retries_strategy="skip",
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
        )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_raise_empty_search_space(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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

    to_be_skipped = check_skip(
        NullContext(),
        model,
        occasion='VISI',
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
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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

    to_be_skipped = check_skip(
        NullContext(),
        model,
        occasion='XAT2',
        allometric_variable=None,
        ignore_datainfo_fallback=False,
        search_space=None,
    )

    assert len(to_be_skipped) == 1


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_iovsearch_missing_occasion_raises(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError, match='Invalid `occasion`'):
        later_input_validation(
            model,
            results=res,
            modeltype='basic_pk',
            administration='oral',
            occasion='XYZ',
            retries_strategy="skip",
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
        )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_ignore_datainfo_fallback(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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
        occasion=None,
        allometric_variable=None,
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
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

    if error != "PASS":
        with pytest.raises(ValueError, match=error):
            later_input_validation(
                model,
                results=res,
                modeltype='basic_pk',
                administration='oral',
                retries_strategy="skip",
                mechanistic_covariates=mechanistic_covariates,
                cl_init=1.0,
                vc_init=10.0,
                mat_init=1.0,
            )
    else:
        # Should not raise any errors
        later_input_validation(
            model,
            results=res,
            modeltype='basic_pk',
            administration='oral',
            retries_strategy="skip",
            mechanistic_covariates=mechanistic_covariates,
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
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


def _load_model(testdata: Path, with_datainfo: bool = False):
    models = testdata / 'nonmem' / 'models'

    # NOTE: We need to make a local copy and read the model locally to avoid
    # reading the .datainfo which contains allometry information we do not want
    # to ignore.

    shutil.copy2(models / 'mox_simulated_normal.csv', '.')
    if with_datainfo:
        shutil.copy2(models / 'mox_simulated_normal.datainfo', '.')
    shutil.copy2(models / 'mox2.mod', '.')
    shutil.copy2(models / 'mox2.ext', '.')
    shutil.copy2(models / 'mox2.lst', '.')
    shutil.copy2(models / 'mox2.phi', '.')

    model = Model.parse_model('mox2.mod')
    model = model.replace(name='start')
    res = read_modelfit_results('mox2.mod')

    # NOTE: Load results directly in DB to skip fitting
    db_tool = default_context(name='amd', ref='amd_dir1')
    db_fit = default_context(name='modelfit', ref=db_tool.path)

    with db_fit.model_database.transaction(model) as txn:
        txn.store_model()
        txn.store_modelfit_results()
        # NOTE: This are needed because currently caching of the results cannot
        # read from the JSON file created above.
        txn.store_local_file(models / 'mox2.ext', 'start.ext')
        txn.store_local_file(models / 'mox2.lst', 'start.lst')
        txn.store_local_file(models / 'mox2.phi', 'start.phi')

    return db_tool, model, res
