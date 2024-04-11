import shutil
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import read_modelfit_results, run_amd
from pharmpy.tools.amd.run import _mechanistic_cov_extraction, validate_input
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.workflows import default_context


def test_invalid_search_space_raises(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata)

        with pytest.raises(
            ValueError,
            match='Invalid `search_space`, could not be parsed:',
        ):
            run_amd(
                model,
                results=res,
                search_space='XYZ',
                retries_strategy="skip",
                path=db.path,
                resume=True,
                cl_init=1.0,
                vc_init=10.0,
                mat_init=1.0,
            )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_most(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata)

        to_be_skipped = validate_input(
            model,
            results=res,
            modeltype='basic_pk',
            administration='oral',
            occasion=None,
            retries_strategy="skip",
            path=db.path,
            resume=True,
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
        )

    assert len(to_be_skipped) == 3


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_raise_allometry(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata, with_datainfo=True)

        with pytest.raises(
            ValueError,
            match='Invalid `allometric_variable`',
        ):
            run_amd(
                model,
                results=res,
                modeltype='basic_pk',
                administration='oral',
                allometric_variable='SJDLKSDJ',
                retries_strategy="skip",
                path=db.path,
                resume=True,
                cl_init=1.0,
                vc_init=10.0,
                mat_init=1.0,
            )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_raise_empty_search_space(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata, with_datainfo=True)

        with pytest.raises(
            ValueError,
            match='`search_space` evaluated to be empty :',
        ):
            run_amd(
                model,
                results=res,
                search_space='LET(CONTINUOUS, [AGE, SJDLKSDJ]); LET(CATEGORICAL, [SEX])',
                modeltype='basic_pk',
                administration='oral',
                retries_strategy="skip",
                path=db.path,
                resume=True,
                cl_init=1.0,
                vc_init=10.0,
                mat_init=1.0,
            )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_covsearch(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata)
        to_be_skipped = validate_input(
            model,
            results=res,
            modeltype='basic_pk',
            administration='oral',
            occasion='VISI',
            allometric_variable='WT',
            retries_strategy="skip",
            path=db.path,
            resume=True,
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
            ignore_datainfo_fallback=True,
        )
    assert "covariates" in to_be_skipped


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_iovsearch_one_occasion(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata, with_datainfo=True)

        to_be_skipped = validate_input(
            model,
            results=res,
            modeltype='basic_pk',
            administration='oral',
            retries_strategy="skip",
            occasion='XAT2',
            path=db.path,
            resume=True,
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
        )

    assert len(to_be_skipped) == 1


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_skip_iovsearch_missing_occasion_raises(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata)

        with pytest.raises(
            ValueError,
            match='Invalid `occasion`',
        ):
            run_amd(
                model,
                results=res,
                modeltype='basic_pk',
                administration='oral',
                occasion='XYZ',
                retries_strategy="skip",
                path=db.path,
                resume=True,
                cl_init=1.0,
                vc_init=10.0,
                mat_init=1.0,
            )


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
def test_ignore_datainfo_fallback(tmp_path, testdata):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata, with_datainfo=True)

        to_be_skipped = validate_input(
            model,
            results=res,
            modeltype='basic_pk',
            administration='oral',
            retries_strategy="skip",
            ignore_datainfo_fallback=True,
            path=db.path,
            resume=True,
            cl_init=1.0,
            vc_init=10.0,
            mat_init=1.0,
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
def test_mechanistic_covariate_option(tmp_path, testdata, mechanistic_covariates, error):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata, with_datainfo=True)

        if error != "PASS":
            with pytest.raises(
                ValueError,
                match=error,
            ):
                validate_input(
                    model,
                    results=res,
                    modeltype='basic_pk',
                    administration='oral',
                    retries_strategy="skip",
                    mechanistic_covariates=mechanistic_covariates,
                    path=db.path,
                    resume=True,
                    cl_init=1.0,
                    vc_init=10.0,
                    mat_init=1.0,
                )
        else:
            # Should not raise any errors
            validate_input(
                model,
                results=res,
                modeltype='basic_pk',
                administration='oral',
                retries_strategy="skip",
                mechanistic_covariates=mechanistic_covariates,
                path=db.path,
                resume=True,
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
    tmp_path, testdata, mechanistic_covariates, expected_mechanistic_ss, expected_filtered_ss
):
    with chdir(tmp_path):
        db, model, res = _load_model(testdata, with_datainfo=True)

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
