import shutil
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import run_amd


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


@pytest.mark.filterwarnings(
    'ignore:.*Adjusting initial estimates to create positive semidefinite omega/sigma matrices.',
    'ignore::UserWarning',
)
def test_amd(tmp_path, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.csv', '.')
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.datainfo', '.')
        input = 'moxo_simulated_amd.csv'
        res = run_amd(
            input,
            modeltype='pk_oral',
            search_space='PERIPHERALS(1)',
            occasion='VISI',
        )

        rundir = tmp_path / 'amd_dir1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        subrundir = [
            'modelfit',
            'modelsearch',
            'iivsearch',
            'ruvsearch',
            'iovsearch',
            'allometry',
            'covsearch',
        ]
        for dir in subrundir:
            dir = rundir / dir
            assert _model_count(dir) >= 1

        assert len(res.summary_tool) == len(subrundir)
        assert len(res.summary_models) >= 1
        assert len(res.summary_individuals_count) >= 1


def test_skip_most(tmp_path, testdata):
    with chdir(tmp_path):
        models = testdata / 'nonmem' / 'models'
        shutil.copy2(models / 'mox_simulated_normal.csv', '.')
        shutil.copy2(models / 'mox2.mod', '.')
        shutil.copy2(models / 'mox2.ext', '.')
        shutil.copy2(models / 'mox2.lst', '.')
        shutil.copy2(models / 'mox2.phi', '.')
        model = Model.create_model('mox2.mod')
        with pytest.warns(Warning) as record:
            res = run_amd(
                model,
                results=model.modelfit_results,
                modeltype='pk_oral',
                order=['iovsearch', 'allometry', 'covariates'],
                continuous=[],
                categorical=[],
                occasion=None,
            )

        assert len(record) == 5

        for warning, match in zip(
            record,
            [
                'IOVsearch will be skipped because occasion is None',
                'Skipping Allometry',
                'Skipping COVsearch',
                'AMDResults.summary_models is None',
                'AMDResults.summary_individuals_count is None',
            ],
        ):
            assert match in str(warning.message)

        assert len(res.summary_tool) == 1
        assert res.summary_models is None
        assert res.summary_individuals_count is None
        assert res.final_model.name == 'start'


def test_skip_iovsearch_one_occasion(tmp_path, testdata):
    with chdir(tmp_path):
        models = testdata / 'nonmem' / 'models'
        shutil.copy2(models / 'mox_simulated_normal.csv', '.')
        shutil.copy2(models / 'mox2.mod', '.')
        shutil.copy2(models / 'mox2.ext', '.')
        shutil.copy2(models / 'mox2.lst', '.')
        shutil.copy2(models / 'mox2.phi', '.')
        model = Model.create_model('mox2.mod')
        with pytest.warns(Warning) as record:
            res = run_amd(
                model,
                results=model.modelfit_results,
                modeltype='pk_oral',
                order=['iovsearch'],
                occasion='XAT2',
            )

        assert len(record) == 3

        for warning, match in zip(
            record,
            [
                'Skipping IOVsearch because there are less than two occasion categories',
                'AMDResults.summary_models is None',
                'AMDResults.summary_individuals_count is None',
            ],
        ):
            assert match in str(warning.message)

        assert len(res.summary_tool) == 1
        assert res.summary_models is None
        assert res.summary_individuals_count is None
        assert res.final_model.name == 'start'


def test_skip_iovsearch_missing_occasion(tmp_path, testdata):
    with chdir(tmp_path):
        models = testdata / 'nonmem' / 'models'
        shutil.copy2(models / 'mox_simulated_normal.csv', '.')
        shutil.copy2(models / 'mox2.mod', '.')
        shutil.copy2(models / 'mox2.ext', '.')
        shutil.copy2(models / 'mox2.lst', '.')
        shutil.copy2(models / 'mox2.phi', '.')
        model = Model.create_model('mox2.mod')
        with pytest.warns(Warning) as record:
            res = run_amd(
                model,
                results=model.modelfit_results,
                modeltype='pk_oral',
                order=['iovsearch'],
                occasion='XYZ',
            )

        assert len(record) == 3

        for warning, match in zip(
            record,
            [
                'Skipping IOVsearch because dataset is missing column "XYZ"',
                'AMDResults.summary_models is None',
                'AMDResults.summary_individuals_count is None',
            ],
        ):
            assert match in str(warning.message)

        assert len(res.summary_tool) == 1
        assert res.summary_models is None
        assert res.summary_individuals_count is None
        assert res.final_model.name == 'start'
