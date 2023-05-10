import shutil

import numpy as np
import pandas as pd
import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import read_model
from pharmpy.tools import fit, run_modelsearch
from pharmpy.workflows import ModelDatabase


def test_exhaustive(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        res = run_modelsearch(
            'ABSORPTION(ZO);PERIPHERALS(1)',
            'exhaustive',
            results=start_model.modelfit_results,
            model=start_model,
        )

        assert len(res.summary_tool) == 4
        assert len(res.summary_models) == 4
        assert len(res.models) == 3
        assert all(model.modelfit_results for model in res.models)
        assert not all(np.isnan(model.modelfit_results.ofv) for model in res.models)
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == 3
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'search_space, no_of_models, last_model_parent_name, model_with_error, ref',
    [
        (
            'ABSORPTION(ZO);PERIPHERALS(1)',
            4,
            'modelsearch_run2',
            'modelsearch_run3',
            ('modelsearch_run2', ['PERIPHERALS(1)', 'VP1 = ']),
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)',
            7,
            'modelsearch_run3',
            'modelsearch_run5',
            ('modelsearch_run3', ['PERIPHERALS(1)', 'VP1 = ']),
        ),
    ],
)
def test_exhaustive_stepwise_basic(
    tmp_path,
    model_count,
    start_model,
    search_space,
    no_of_models,
    last_model_parent_name,
    model_with_error,
    ref,
):
    with chdir(tmp_path):
        res = run_modelsearch(
            search_space,
            'exhaustive_stepwise',
            results=start_model.modelfit_results,
            model=start_model,
        )

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

        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_models
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()

        db: ModelDatabase = res.tool_database.model_database
        model_name, code_ref = ref
        path = db.retrieve_file(model_name, f'{model_name}.mod')
        with open(path, 'r') as fh:
            model_code = fh.read()
            assert all(code in model_code for code in code_ref)


@pytest.mark.parametrize(
    'search_space, iiv_strategy, no_of_models, no_of_added_etas',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 'add_diagonal', 4, 2),
        ('ABSORPTION(ZO);PERIPHERALS(1)', 'fullblock', 4, 2),
        ('PERIPHERALS(1);LAGTIME()', 'absorption_delay', 4, 1),
    ],
)
def test_exhaustive_stepwise_iiv_strategies(
    tmp_path,
    model_count,
    start_model,
    search_space,
    iiv_strategy,
    no_of_models,
    no_of_added_etas,
):
    with chdir(tmp_path):
        res = run_modelsearch(
            search_space,
            'exhaustive_stepwise',
            iiv_strategy=iiv_strategy,
            results=start_model.modelfit_results,
            model=start_model,
        )

        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.models) == no_of_models
        model_last = res.models[no_of_models - 1]
        assert (
            len(model_last.random_variables.etas.names)
            - len(start_model.random_variables.etas.names)
            == no_of_added_etas
        )
        assert model_last.modelfit_results

        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_models
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


# def test_exhaustive_stepwise_start_model_not_fitted(tmp_path, model_count, start_model):
#    with chdir(tmp_path):
#        start_model = start_model.copy()
#        start_model.name = 'start_model_copy'
#        start_model.modelfit_results = None

#        search_space = 'ABSORPTION(ZO);PERIPHERALS(1)'
#        with pytest.warns(UserWarning, match='Could not update'):
#            res = run_modelsearch(
#                search_space,
#                'exhaustive_stepwise',
#                results=start_model.modelfit_results,
#                model=start_model,
#            )

#        assert len(res.summary_tool) == 5
#        assert len(res.summary_models) == 5
#        assert res.summary_tool['dbic'].isnull().values.all()
#        assert len(res.models) == 4
#        rundir = tmp_path / 'modelsearch_dir1'
#        assert rundir.is_dir()
#        assert model_count(rundir) == 4


def test_exhaustive_stepwise_peripheral_upper_limit(tmp_path, start_model):
    with chdir(tmp_path):
        res = run_modelsearch(
            'PERIPHERALS(1)',
            'exhaustive_stepwise',
            results=start_model.modelfit_results,
            model=start_model,
        )

        assert ',999999) ; POP_QP1' in res.models[0].model_code
        assert ',999999) ; POP_VP1' in res.models[0].model_code


def test_summary_individuals(tmp_path, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        m = read_model('pheno_real.mod')
        start_res = fit(m)
        m = m.replace(modelfit_results=start_res)
        res = run_modelsearch(
            model=m,
            results=m.modelfit_results,
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
            # Cannot check that all are non-na because some model runs fail
            assert summary[column].notna().any()
        assert summary['dofv_vs_parent'].equals(
            summary.apply(
                lambda row: summary.loc[(row['parent_model'], row.name[1])]['ofv'] - row['ofv'],
                axis=1,
            )
        )
