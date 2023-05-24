import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import run_amd
from pharmpy.workflows import default_tool_database


def test_invalid_search_space_raises(tmp_path, testdata):
    with chdir(tmp_path):
        db, model = _load_model(testdata)

        with pytest.raises(
            ValueError,
            match='Invalid `search_space`, could not be parsed:',
        ):
            run_amd(
                model,
                results=model.modelfit_results,
                search_space='XYZ',
                path=db.path,
                resume=True,
            )


def test_skip_most(tmp_path, testdata):
    with chdir(tmp_path):
        db, model = _load_model(testdata)

        with _record_warnings() as record:
            res = run_amd(
                model,
                results=model.modelfit_results,
                modeltype='pk_oral',
                order=['iovsearch', 'allometry', 'covariates'],
                occasion=None,
                path=db.path,
                resume=True,
            )

        _validate_record(
            record,
            [
                'IOVsearch will be skipped because occasion is None',
                'Allometry will most likely be skipped',
                'COVsearch will most likely be skipped',
                'Skipping Allometry',
                'Skipping COVsearch',
                'AMDResults.summary_models is None',
                'AMDResults.summary_individuals_count is None',
            ],
        )

        assert len(res.summary_tool) == 1
        assert res.summary_models is None
        assert res.summary_individuals_count is None
        assert res.final_model == 'start'


def test_raise_allometry(tmp_path, testdata):
    with chdir(tmp_path):
        db, model = _load_model(testdata, with_datainfo=True)

        with pytest.raises(
            ValueError,
            match='Invalid `allometric_variable`',
        ):
            run_amd(
                model,
                results=model.modelfit_results,
                modeltype='pk_oral',
                order=['allometry'],
                allometric_variable='SJDLKSDJ',
                path=db.path,
                resume=True,
            )


def test_raise_covsearch(tmp_path, testdata):
    with chdir(tmp_path):
        db, model = _load_model(testdata, with_datainfo=True)

        with pytest.raises(
            ValueError,
            match='Invalid `search_space` because of invalid covariate .* got `SJDLKSDJ`',
        ):
            run_amd(
                model,
                results=model.modelfit_results,
                search_space='LET(CONTINUOUS, [AGE, SJDLKSDJ]); LET(CATEGORICAL, [SEX])',
                modeltype='pk_oral',
                order=['covariates'],
                path=db.path,
                resume=True,
            )


def test_skip_covsearch(tmp_path, testdata):
    with chdir(tmp_path):
        db, model = _load_model(testdata, with_datainfo=True)

        with _record_warnings() as record:
            res = run_amd(
                model,
                results=model.modelfit_results,
                search_space='LET(CONTINUOUS, []); LET(CATEGORICAL, [])',
                modeltype='pk_oral',
                order=['covariates'],
                path=db.path,
                resume=True,
            )
        _validate_record(
            record,
            [
                'COVsearch will most likely be skipped',
                'Skipping COVsearch',
                'AMDResults.summary_models is None',
                'AMDResults.summary_individuals_count is None',
            ],
        )

        assert len(res.summary_tool) == 1
        assert res.summary_models is None
        assert res.summary_individuals_count is None
        assert res.final_model == 'start'


def test_skip_iovsearch_one_occasion(tmp_path, testdata):
    with chdir(tmp_path):
        db, model = _load_model(testdata)

        with _record_warnings() as record:
            res = run_amd(
                model,
                results=model.modelfit_results,
                modeltype='pk_oral',
                order=['iovsearch'],
                occasion='XAT2',
                path=db.path,
                resume=True,
            )

        _validate_record(
            record,
            [
                'Skipping IOVsearch because there are less than two occasion categories',
                'AMDResults.summary_models is None',
                'AMDResults.summary_individuals_count is None',
            ],
        )

        assert len(res.summary_tool) == 1
        assert res.summary_models is None
        assert res.summary_individuals_count is None
        assert res.final_model == 'start'


def test_skip_iovsearch_missing_occasion_raises(tmp_path, testdata):
    with chdir(tmp_path):
        db, model = _load_model(testdata)

        with pytest.raises(
            ValueError,
            match='Invalid `occasion`',
        ):
            run_amd(
                model,
                results=model.modelfit_results,
                modeltype='pk_oral',
                order=['iovsearch'],
                occasion='XYZ',
                path=db.path,
                resume=True,
            )


def _load_model(testdata: Path, with_datainfo: bool = False):
    models = testdata / 'nonmem' / 'models'

    # NOTE We need to make a local copy and read the model locally to avoid
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

    # NOTE Load results directly in DB to skip fitting
    db_tool = default_tool_database(toolname='amd', path='amd_dir1')
    db_fit = default_tool_database(toolname='modelfit', path=db_tool.path / 'modelfit')

    with db_fit.model_database.transaction(model) as txn:
        txn.store_model()
        txn.store_modelfit_results()
        # NOTE This are needed because currently caching of the results cannot
        # read from the JSON file created above.
        txn.store_local_file(models / 'mox2.ext', 'start.ext')
        txn.store_local_file(models / 'mox2.lst', 'start.lst')
        txn.store_local_file(models / 'mox2.phi', 'start.phi')

    return db_tool, model


@contextmanager
def _record_warnings():
    with pytest.warns(Warning) as record:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                module='distributed',
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                module='distributed',
                category=ResourceWarning,
            )
            warnings.filterwarnings(
                "ignore",
                module='distributed',
                category=RuntimeWarning,
            )
            yield record


def _validate_record(record, expected):
    assert len(record) == len(expected)

    for warning, match in zip(record, expected):
        assert match in str(warning.message)
