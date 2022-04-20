import shutil
import warnings

import numpy as np
import pytest

from pharmpy import Model
from pharmpy.modeling import add_iiv, fit, read_model, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_exhaustive(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'
        res = run_tool(
            'modelsearch', 'ABSORPTION(ZO);PERIPHERALS(1)', 'exhaustive', model=model_start
        )

        assert len(res.summary_tool) == 4
        assert len(res.summary_models) == 4
        assert len(res.models) == 3
        assert res.best_model.name == 'modelsearch_candidate3'
        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 5
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'mfl, no_of_models, best_model_name, last_model_parent_name',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 4, 'modelsearch_candidate2', 'modelsearch_candidate2'),
        ('ABSORPTION(ZO);ELIMINATION(ZO)', 4, 'modelsearch_candidate1', 'modelsearch_candidate2'),
        ('ABSORPTION(ZO);TRANSITS(1)', 2, 'modelsearch_candidate2', 'mox2'),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)',
            7,
            'modelsearch_candidate6',
            'modelsearch_candidate3',
        ),
        ('LAGTIME();TRANSITS(1)', 2, 'modelsearch_candidate2', 'mox2'),
        ('ABSORPTION(ZO);TRANSITS(3, *)', 3, 'modelsearch_candidate2', 'mox2'),
    ],
)
def test_exhaustive_stepwise_basic(
    tmp_path, testdata, mfl, no_of_models, best_model_name, last_model_parent_name
):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'
        res = run_tool('modelsearch', mfl, 'exhaustive_stepwise', model=model_start)

        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.models) == no_of_models
        assert res.best_model.name == best_model_name

        assert res.models[0].parent_model == 'mox2'
        assert res.models[-1].parent_model == last_model_parent_name
        if last_model_parent_name != 'mox2':
            last_model_features = res.summary_tool.loc[res.models[-1].name]['features']
            parent_model_features = res.summary_tool.loc[last_model_parent_name]['features']
            assert last_model_features[: len(parent_model_features)] == parent_model_features

        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 2
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'mfl, iiv_strategy, no_of_models, best_model_name, no_of_added_etas',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 1, 4, 'modelsearch_candidate2', 2),
        ('ABSORPTION(ZO);ELIMINATION(ZO)', 1, 4, 'modelsearch_candidate1', 1),
        ('ABSORPTION(ZO);ELIMINATION(MIX-FO-MM)', 1, 4, 'modelsearch_candidate1', 2),
        ('ABSORPTION(ZO);PERIPHERALS([1, 2])', 1, 8, 'modelsearch_candidate5', 4),
        ('LAGTIME();TRANSITS(1)', 1, 2, 'modelsearch_candidate2', 1),
        ('ABSORPTION(ZO);PERIPHERALS(1)', 2, 4, 'modelsearch_candidate4', 2),
        ('PERIPHERALS(1);LAGTIME()', 3, 4, 'modelsearch_candidate3', 1),
    ],
)
def test_exhaustive_stepwise_add_iivs(
    tmp_path,
    testdata,
    mfl,
    iiv_strategy,
    no_of_models,
    best_model_name,
    no_of_added_etas,
):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            res = run_tool(
                'modelsearch',
                mfl,
                'exhaustive_stepwise',
                iiv_strategy=iiv_strategy,
                model=model_start,
            )

        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.models) == no_of_models
        assert res.best_model.name == best_model_name
        model_last = res.models[no_of_models - 1]
        assert (
            len(model_last.random_variables.etas) - len(model_start.random_variables.etas)
            == no_of_added_etas
        )
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 2
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


def test_exhaustive_stepwise_already_fit(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'

        fit(model_start)

        mfl = 'ABSORPTION(ZO);PERIPHERALS(1)'
        res = run_tool('modelsearch', mfl, 'exhaustive_stepwise', model=model_start)

        assert len(res.summary_tool) == 5
        assert len(res.summary_models) == 5
        assert len(res.models) == 4
        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 5
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


def test_exhaustive_stepwise_start_model_fail(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'

        add_iiv(model_start, 'V', 'incorrect_syntax')

        mfl = 'ABSORPTION(ZO);PERIPHERALS(1)'
        with pytest.warns(UserWarning, match='Could not update'):
            res = run_tool('modelsearch', mfl, 'exhaustive_stepwise', model=model_start)

        assert len(res.summary_tool) == 5
        assert len(res.summary_models) == 5
        assert res.summary_tool['dofv'].isnull().values.all()
        assert len(res.models) == 4
        assert all(model.modelfit_results is None for model in res.models)
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 6


@pytest.mark.parametrize(
    'mfl, no_of_models, best_model_name, last_model_parent_name',
    [
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS(1)',
            12,
            'modelsearch_candidate9',
            'modelsearch_candidate9',
        ),
    ],
)
def test_reduced_stepwise(
    tmp_path, testdata, mfl, no_of_models, best_model_name, last_model_parent_name
):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'
        res = run_tool('modelsearch', mfl, 'reduced_stepwise', model=model_start)

        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.models) == no_of_models
        assert res.best_model.name == best_model_name

        assert res.models[0].parent_model == 'mox2'
        assert res.models[-1].parent_model == last_model_parent_name
        if last_model_parent_name != 'mox2':
            last_model_features = res.summary_tool.loc[res.models[-1].name]['features']
            parent_model_features = res.summary_tool.loc[last_model_parent_name]['features']
            assert last_model_features[: len(parent_model_features)] == parent_model_features

        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 2
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


def test_summary_individuals(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        m = read_model('pheno_real.mod')
        fit(m)
        res = run_tool(
            'modelsearch',
            model=m,
            mfl='ABSORPTION(ZO);PERIPHERALS([1, 2])',
            algorithm='reduced_stepwise',
        )
        summary = res.summary_individuals
        columns = (
            'parent_model',
            'outliers_fda',
            'ofv',
            'dofv_vs_parent',
            'predicted_dofv',
            'predicted_residual',
        )
        assert summary is not None
        assert tuple(summary.columns) == columns
        for column in columns:
            assert summary[column].notna().any()
        assert summary['dofv_vs_parent'].equals(
            summary.apply(
                lambda row: summary.loc[(row['parent_model'], row.name[1])]['ofv'] - row['ofv'],
                axis=1,
            )
        )
