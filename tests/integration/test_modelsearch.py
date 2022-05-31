import shutil
from pathlib import Path

import numpy as np
import pytest

from pharmpy.modeling import fit, read_model, run_modelsearch
from pharmpy.utils import TemporaryDirectoryChanger


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


def test_exhaustive(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_modelsearch('ABSORPTION(ZO);PERIPHERALS(1)', 'exhaustive', model=start_model)

        assert len(res.summary_tool) == 4
        assert len(res.summary_models) == 4
        assert len(res.models) == 3
        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == 3
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'search_space, no_of_models, last_model_parent_name, model_with_error',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 4, 'modelsearch_candidate2', 'modelsearch_candidate3'),
        # FIXME: Warning after setting TOL=9
        # ('ABSORPTION(ZO);ELIMINATION(ZO)', 4, 'modelsearch_candidate1', 'modelsearch_candidate2'),
        ('ABSORPTION(ZO);TRANSITS(1)', 2, 'mox2', ''),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)',
            7,
            'modelsearch_candidate3',
            'modelsearch_candidate5',
        ),
        ('LAGTIME();TRANSITS(1)', 2, 'mox2', ''),
        ('ABSORPTION(ZO);TRANSITS(3, *)', 3, 'mox2', ''),
    ],
)
def test_exhaustive_stepwise_basic(
    tmp_path, start_model, search_space, no_of_models, last_model_parent_name, model_with_error
):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_modelsearch(search_space, 'exhaustive_stepwise', model=start_model)

        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.models) == no_of_models
        assert res.models[-1].modelfit_results

        assert res.models[0].parent_model == 'mox2'
        assert res.models[-1].parent_model == last_model_parent_name
        if last_model_parent_name != 'mox2':
            last_model_features = res.summary_tool.loc[res.models[-1].name]['description']
            parent_model_features = res.summary_tool.loc[last_model_parent_name]['description']
            assert last_model_features[: len(parent_model_features)] == parent_model_features

        if model_with_error:
            assert model_with_error in res.summary_errors.index.get_level_values('model')

        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_models
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    'search_space, iiv_strategy, no_of_models, no_of_added_etas',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 'diagonal', 4, 2),
        ('ABSORPTION(ZO);ELIMINATION(ZO)', 'diagonal', 4, 1),
        ('ABSORPTION(ZO);ELIMINATION(MIX-FO-MM)', 'diagonal', 4, 2),
        ('ABSORPTION(ZO);PERIPHERALS([1, 2])', 'diagonal', 8, 4),
        ('LAGTIME();TRANSITS(1)', 'diagonal', 2, 1),
        ('ABSORPTION(ZO);PERIPHERALS(1)', 'fullblock', 4, 2),
        ('PERIPHERALS(1);LAGTIME()', 'absorption_delay', 4, 1),
    ],
)
def test_exhaustive_stepwise_add_iivs(
    tmp_path,
    start_model,
    search_space,
    iiv_strategy,
    no_of_models,
    no_of_added_etas,
):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_modelsearch(
            search_space,
            'exhaustive_stepwise',
            iiv_strategy=iiv_strategy,
            model=start_model,
        )

        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.models) == no_of_models
        model_last = res.models[no_of_models - 1]
        assert (
            len(model_last.random_variables.etas) - len(start_model.random_variables.etas)
            == no_of_added_etas
        )
        assert model_last.modelfit_results

        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_models
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


def test_exhaustive_stepwise_start_model_not_fitted(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        start_model = start_model.copy()
        start_model.name = 'start_model_copy'
        start_model.modelfit_results = None

        search_space = 'ABSORPTION(ZO);PERIPHERALS(1)'
        with pytest.warns(UserWarning, match='Could not update'):
            res = run_modelsearch(search_space, 'exhaustive_stepwise', model=start_model)

        assert len(res.summary_tool) == 5
        assert len(res.summary_models) == 5
        assert res.summary_tool['dbic'].isnull().values.all()
        assert len(res.models) == 4
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == 4


def test_exhaustive_stepwise_peripheral_upper_limit(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_modelsearch('PERIPHERALS(1)', 'exhaustive_stepwise', model=start_model)

        assert ',999999) ; POP_QP1' in res.models[0].model_code
        assert ',999999) ; POP_VP1' in res.models[0].model_code


def test_summary_individuals(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        m = read_model('pheno_real.mod')
        fit(m)
        res = run_modelsearch(
            model=m,
            search_space='ABSORPTION(ZO);PERIPHERALS([1, 2])',
            algorithm='reduced_stepwise',
        )
        summary = res.summary_individuals
        columns = (
            'description',
            'parent_model',
            'outlier_count',
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
